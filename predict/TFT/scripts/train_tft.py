"""
TFT Training Script - Final (Interpretation 구조 개선 버전)

✔ mean pooling 제거
✔ sample / date / horizon 단위 interpretation 저장
✔ quantile_loss ON/OFF 모두 대응
✔ 시점 중요도: t-20 ~ t-1 실제 attention
✔ feature 중요도: timestep 평균만 (sample 단위)

저장 구조:
checkpoints/TFT_corn_fold0_h1-5-10-20/
├── best_model.pt
├── visualizations/
│   └── loss_curve.png
└── interpretations/
    └── fold0/
        ├── h1/
        │   └── YYYY-MM-DD.npz
        ├── h5/
        ├── h10/
        └── h20/
"""

import os
import sys
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import numpy as np
import tyro
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
PREDICT_ROOT = MODEL_ROOT.parent
MODEL_SRC = MODEL_ROOT / "src"
for _path in (MODEL_SRC, PREDICT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_ROOT = PREDICT_ROOT.parent

from configs.tft_config import TFTConfig as TrainConfig
from shared.utils.set_seed import set_seed
from data.dataset_tft import TFTDataLoader
from shared.data.bigquery_loader import load_price_table, load_news_features_bq
from models.TFT import TemporalFusionTransformer, QuantileLoss
from engine.trainer_tft import train
from shared.utils.visualization import save_loss_curve


def _strip_quotes(value: str) -> str:
    if not value:
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = _strip_quotes(val.strip())
    return env


def _resolve_price_bq_settings(config: TrainConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("PRICE_BIGQUERY_PROJECT") or config.bq_project_id,
        "dataset_id": _get("PRICE_BIGQUERY_DATASET") or config.bq_dataset_id,
        "credentials_path": _get("PRICE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def _resolve_article_bq_settings(config: TrainConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("ARTICLE_BIGQUERY_PROJECT") or config.bq_news_project_id,
        "dataset_id": _get("ARTICLE_BIGQUERY_DATASET") or config.bq_news_dataset_id,
        "credentials_path": _get("ARTICLE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "daily_table": _get("TABLE_DAILY_SUMMARY") or config.bq_news_table,
    }


def _fetch_latest_inference_date(
    project_id: str,
    dataset_id: str,
    table: str,
    credentials_path: Optional[str] = None,
) -> Optional[str]:
    if not project_id or not dataset_id or not table:
        return None
    try:
        from google.cloud import bigquery
    except Exception:
        return None

    creds = None
    if credentials_path:
        try:
            cred_path = Path(credentials_path)
            if cred_path.is_file():
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(str(cred_path))
        except Exception:
            creds = None

    client = bigquery.Client(project=project_id, credentials=creds)

    def _run_query(field: str) -> Optional[str]:
        query = (
            f"SELECT MAX({field}) AS max_date "
            f"FROM `{project_id}.{dataset_id}.{table}`"
        )
        job = client.query(query)
        rows = list(job.result())
        if not rows:
            return None
        max_date = rows[0].get("max_date")
        if max_date is None:
            return None
        return str(max_date)[:10]

    try:
        date_str = _run_query("trade_date")
    except Exception:
        date_str = None
    if date_str:
        return date_str
    try:
        return _run_query("time")
    except Exception:
        return None

def build_trading_index(price_df):
    # price_df는 data_loader.data (merge된 DF)
    d = pd.to_datetime(price_df["time"]).dt.date.tolist()
    return d

def get_window_dates(trading_dates, base_date_str, seq_length):
    '''
    윈도우 날짜 생성
    
    Args:
        trading_dates: 전체 거래일 리스트
        base_date_str: 입력 윈도우의 마지막 날짜 (t)
        seq_length: 윈도우 길이
    
    Returns:
        [t-(seq_length-1), ..., t] 날짜 리스트 (길이 = seq_length)
    
    Example:
        base_date = "2020-11-23"
        seq_length = 20
        → 윈도우: [2020-10-28, ..., 2020-11-23] (20개 날짜)
        → 파일명: 2020-11-23.npz (base_date)
    '''
    base = pd.to_datetime(base_date_str).date()
    
    # base_date의 인덱스 찾기
    try:
        base_idx = trading_dates.index(base)
    except ValueError:
        # base_date가 거래일이 아닌 경우 가장 가까운 이전 거래일 사용
        valid_dates = [d for d in trading_dates if d <= base]
        if not valid_dates:
            return []
        base = max(valid_dates)
        base_idx = trading_dates.index(base)
    
    # ✅ 윈도우: [base_idx - seq_length + 1 : base_idx + 1]
    # 즉, t-(seq_length-1) ~ t까지 (base 포함)
    if base_idx < seq_length - 1:
        return []
    
    window_dates = trading_dates[base_idx - seq_length + 1 : base_idx + 1]
    return window_dates

def get_target_date(trading_dates, base_date_str, horizon):
    base = pd.to_datetime(base_date_str).date()
    if base not in trading_dates:
        return None
    idx = trading_dates.index(base)
    tgt = idx + horizon  # (t-1)+h = t+h-1
    if tgt >= len(trading_dates):
        return None
    return str(trading_dates[tgt])

def get_base_date_from_h1_target(trading_dates, h1_target_date_str):
    """
    valid_dates가 'h1의 target_date(t)'일 때,
    base_date는 그 이전 거래일(t-1)로 복원한다.
    """
    tgt = pd.to_datetime(h1_target_date_str).date()
    if tgt not in trading_dates:
        return None
    idx = trading_dates.index(tgt)
    base_idx = idx - 1  # h1 기준 base = t-1
    if base_idx < 0:
        return None
    return str(trading_dates[base_idx])

def _run_single(config: TrainConfig):
    set_seed(config.seed)

    h_tag = "h" + "-".join(map(str, config.horizons))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    print(f"[DEBUG] horizons = {config.horizons} → {h_tag}")

    base_checkpoint_dir = Path(config.checkpoint_dir)
    if base_checkpoint_dir.name == "checkpoints" and config.data_source == "bigquery":
        bq_settings = _resolve_price_bq_settings(config)
        latest_date = _fetch_latest_inference_date(
            project_id=bq_settings.get("project_id"),
            dataset_id=bq_settings.get("dataset_id"),
            table=config.bq_inference_table,
            credentials_path=bq_settings.get("credentials_path"),
        )
        if latest_date:
            base_checkpoint_dir = base_checkpoint_dir / f"{latest_date}_{config.target_commodity}_tft"
            print(f"✓ Using checkpoint root: {base_checkpoint_dir}")
        else:
            print(
                "⚠️  Could not resolve latest inference trade_date; "
                f"using checkpoint_dir as-is: {base_checkpoint_dir}"
            )

    train_price_path = os.path.join(config.data_dir, "train_price.csv")
    feature_path = os.path.join(
        config.data_dir,
        f"preprocessing/{config.target_commodity}_feature_engineering.csv"
    )
    news_path = os.path.join(config.data_dir, "news_features.csv")
    news_source = news_path
    price_source = feature_path
    if config.data_source == "bigquery":
        price_bq = _resolve_price_bq_settings(config)
        if price_bq.get("credentials_path") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = price_bq["credentials_path"]
        price_source = load_price_table(
            project_id=price_bq.get("project_id") or config.bq_project_id,
            dataset_id=price_bq.get("dataset_id") or config.bq_dataset_id,
            table=config.bq_train_table,
            commodity=config.target_commodity,
        )
    else:
        if os.path.exists(train_price_path):
            price_source = train_price_path
    if getattr(config, "news_source", "csv") == "bigquery":
        article_bq = _resolve_article_bq_settings(config)
        if article_bq.get("credentials_path") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = article_bq["credentials_path"]
        news_source = load_news_features_bq(
            project_id=article_bq.get("project_id") or config.bq_news_project_id,
            dataset_id=article_bq.get("dataset_id") or config.bq_news_dataset_id,
            table=article_bq.get("daily_table") or config.bq_news_table,
            commodity=config.target_commodity,
            credentials_path=article_bq.get("credentials_path"),
        )
    split_file = config.split_file
    if "{commodity}" in split_file:
        split_file = split_file.format(commodity=config.target_commodity)
    split_path = Path(split_file)
    if not split_path.is_absolute():
        candidate = PREDICT_ROOT / split_file
        if candidate.exists():
            split_path = candidate
        else:
            split_path = (Path(config.data_dir) / split_file).resolve()
    else:
        split_path = split_path.resolve()

    # --------------------
    # Data loader
    # --------------------
    data_loader = TFTDataLoader(
        price_data_path=price_source,
        news_data_path=news_source,
        split_file=str(split_path),
        seq_length=config.seq_length,
        horizons=config.horizons,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    print(f"✓ Total features: {data_loader.X.shape[-1]}")
    print(f"✓ Horizons: {config.horizons}")

    for fold in config.fold:
        print(f"\n{'='*60}")
        print(f"Window w{config.seq_length} | Fold {fold}")
        print(f"{'='*60}")

        # --------------------
        # Directories
        # --------------------
        if getattr(config, "checkpoint_layout", "legacy") == "simple":
            # Keep per-window outputs separated when running multiple seq_length values.
            fold_dir = base_checkpoint_dir / f"w{config.seq_length}"
            if len(config.fold) > 1:
                fold_dir = fold_dir / f"fold_{fold}"
        else:
            fold_dir = base_checkpoint_dir / f"TFT_{config.target_commodity}_fold{fold}_{h_tag}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        viz_dir = None
        if config.save_train_visualizations:
            viz_dir = fold_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

        interp_root = None
        if config.compute_feature_importance or config.compute_temporal_importance:
            interp_root = fold_dir / "interpretations" / f"fold{fold}"
            interp_root.mkdir(parents=True, exist_ok=True)

        pred_dir = None
        pred_rows = []
        if config.save_val_predictions:
            pred_dir = fold_dir / "predictions"
            pred_dir.mkdir(exist_ok=True)

        # --------------------
        # Loaders  (★ 반드시 여기)
        # --------------------
        train_loader, valid_loader, valid_dates = data_loader.get_fold_loaders(
            fold,
            scale_x=config.scale_x,
            scale_y=config.scale_y,
            eps=config.scaler_eps
        )
        trading_dates = build_trading_index(data_loader.data)

        # --------------------
        # Model
        # --------------------
        model = TemporalFusionTransformer(
            num_features=data_loader.X.shape[-1],
            num_horizons=len(config.horizons),
            hidden_dim=config.hidden_dim,
            lstm_layers=config.num_layers,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
            use_variable_selection=config.use_variable_selection,
            quantiles=config.quantiles if config.quantile_loss else None,
            news_projection_dim = config.news_projection_dim,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        criterion = (
            QuantileLoss(config.quantiles)
            if config.quantile_loss
            else nn.MSELoss()
        )

        # --------------------
        # Train
        # --------------------
        scaler_dict = data_loader.fold_scalers.get(fold, None)
        y_scaler = None
        if scaler_dict and scaler_dict.get("scale_y", False):
            # SimpleYScaler 객체 추출
            y_scaler = scaler_dict.get("y_scaler_obj", None)
        
        model, train_hist, valid_hist, best_metrics, best_epoch, best_val_loss = train(
            model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            device,
            num_epochs=config.epochs,
            patience=config.early_stopping_patience,
            horizons=config.horizons,
            y_scaler=y_scaler
        )

        # --------------------
        # Save checkpoint
        # --------------------
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_metrics": best_metrics,
                "best_epoch": best_epoch,
                "config": vars(config),
            },
            fold_dir / "best_model.pt",
        )

        scaler = data_loader.fold_scalers.get(fold, None)
        if scaler is not None:
            np.savez(
                fold_dir / "scaler.npz",
                scale_x=scaler.get("scale_x", False),
                scale_y=scaler.get("scale_y", False),
                x_mean=scaler.get("x_mean", None),
                x_std=scaler.get("x_std", None),
                y_mean=scaler.get("y_mean", None),
                y_std=scaler.get("y_std", None),
                feature_names=scaler.get("feature_names", None),
            )
            print(f"✓ Saved scaler: {fold_dir / 'scaler.npz'}")
        
        # --------------------
        # Save metrics (per fold)
        # --------------------
        metrics_path = fold_dir / "val_metrics.json"
        if best_metrics is not None:
            payload = {
                "commodity": config.target_commodity,
                "fold": fold,
                "horizons": config.horizons,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "overall": {
                    "MAE": float(best_metrics["mae_overall"]),
                    "RMSE": float(best_metrics["rmse_overall"]),
                    "DA": float(best_metrics["da_overall"]),
                    "R2": float(best_metrics["r2_overall"]),
                },
                "per_horizon": {
                    str(h): {
                        "MAE": float(best_metrics[f"mae_h{i}"]),
                        "RMSE": float(best_metrics[f"rmse_h{i}"]),
                        "DA": float(best_metrics[f"da_h{i}"]),
                        "R2": float(best_metrics[f"r2_h{i}"]),
                    }
                    for i, h in enumerate(config.horizons)
                }
            }
            with open(metrics_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"✓ Saved val metrics: {metrics_path}")

        # --------------------
        # Update global training summary (optional)
        # --------------------
        if getattr(config, "save_training_summary", False):
            summary_path = Path(config.output_dir) / "training_summary.json"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    all_summaries = json.load(f)
            else:
                all_summaries = {}

            commodity_key = config.target_commodity
            all_summaries.setdefault(commodity_key, {})

            fold_key = f"fold_{fold}"
            entry = {
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_val_loss),
                "final_train_loss": float(train_hist[-1]),
                "final_valid_loss": float(valid_hist[-1]),
            }
            if best_metrics is not None:
                entry.update({k: float(v) for k, v in best_metrics.items()})

            all_summaries[commodity_key][fold_key] = entry

            # best_fold 갱신
            best_fold = None
            best_loss = float("inf")
            for k, v in all_summaries[commodity_key].items():
                if not k.startswith("fold_"):
                    continue
                if v.get("best_valid_loss", float("inf")) < best_loss:
                    best_loss = v["best_valid_loss"]
                    best_fold = k
            if best_fold is not None:
                all_summaries[commodity_key]["best_fold"] = best_fold

            with open(summary_path, "w") as f:
                json.dump(all_summaries, f, indent=2)

            print(f"✓ Updated training summary: {summary_path}")


        if viz_dir is not None:
            save_loss_curve(
                train_hist,
                valid_hist,
                str(viz_dir),
                "loss_curve.png",
            )

        # ======================================================
        # Interpretation (NO MEAN POOLING)
        # ======================================================
        if not (config.compute_feature_importance or config.compute_temporal_importance):
            continue

        print("\n📊 Computing interpretations (sample-wise)...")

        model.eval()

        scaler = data_loader.fold_scalers.get(fold, None)
        y_mean = scaler.get("y_mean", None) if scaler is not None else None
        y_std  = scaler.get("y_std", None)  if scaler is not None else None
        use_y_scale = (scaler is not None and scaler.get("scale_y", False) and y_mean is not None and y_std is not None)

        sample_ptr = 0

        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                outputs = model(x_val, return_attention=True)
                preds = outputs["predictions"]

                var_w = outputs.get("variable_importance", None)  # [B, T, F]
                attn_w = outputs.get("attention_weights", None)  # [B, H, T, T]
                attn_h = outputs.get("horizon_attention_weights", None)  # [B, heads, H, T]

                B = x_val.size(0)

                for i in range(B):
                        # valid_dates는 'h1 target_date(t)'라고 가정
                    base_date = str(valid_dates[sample_ptr])[:10]
                    h1_target_date = get_target_date(trading_dates, base_date, 1)

                    # base_date가 없거나 lookback이 부족하면 skip
                    if base_date is None:
                        sample_ptr += 1
                        continue

                    window_dates = get_window_dates(trading_dates, base_date, config.seq_length)
                    if window_dates is None:
                        sample_ptr += 1
                        continue

                    sample_ptr += 1

                    row = {"date": base_date}
                    def inv_y(val, h_idx):
                        if not use_y_scale:
                            return float(val)
                        return float(val * y_std[h_idx] + y_mean[h_idx])
                    
                    for h_idx, horizon in enumerate(config.horizons):
                        row[f"target_h{horizon}"] = inv_y(y_val[i, h_idx].item(), h_idx)
                        if config.quantile_loss:
                            if 0.5 in config.quantiles:
                                q_idx = config.quantiles.index(0.5)
                            else:
                                q_idx = len(config.quantiles) // 2
                            row[f"pred_h{horizon}"] = inv_y(preds[i, h_idx, q_idx].item(), h_idx)
                        else:
                            row[f"pred_h{horizon}"] = inv_y(preds[i, h_idx].item(), h_idx)
                    pred_rows.append(row)

                    # ===== Horizon별 저장 루프 =====
                    for h_idx, horizon in enumerate(config.horizons):
                        save_dir = interp_root / f"h{horizon}"
                        save_dir.mkdir(parents=True, exist_ok=True)

                        data = {
                            "base_date": base_date,
                            "horizon": horizon,
                            "h1_target_date": h1_target_date,
                        }

                        if window_dates is not None:
                            data["window_dates"] = np.array([str(d) for d in window_dates], dtype=object)

                        target_date = get_target_date(trading_dates, base_date, horizon)
                        data["target_date"] = target_date if target_date is not None else base_date

                        # ========================================
                        # ✅ Temporal Importance (Horizon-Specific!)
                        # ========================================
                        temporal_imp = None
                        if attn_h is not None and config.compute_temporal_importance:
                            # Horizon-specific attention
                            attn_sample = attn_h[i].cpu().numpy()  # [heads, H, T]
                            attn_mean = attn_sample.mean(axis=0)   # [H, T]
                            
                            # ✅ 이 horizon의 temporal importance
                            temporal_imp = attn_mean[h_idx]  # [T]
                            data["temporal_importance"] = temporal_imp
                            
                            # ✅ Self-attention matrix도 저장 (heatmap용)
                            if attn_w is not None:
                                attn_self = attn_w[i].cpu().numpy()  # [heads, T, T]
                                attn_self_mean = attn_self.mean(axis=0)  # [T, T]
                                data["attention_matrix"] = attn_self_mean
                        
                        elif attn_w is not None and config.compute_temporal_importance:
                            # Fallback: horizon attention 없으면 self-attention 사용
                            attn_sample = attn_w[i].cpu().numpy()  # [heads, T, T]
                            attn_mean = attn_sample.mean(axis=0)   # [T, T]
                            temporal_imp = attn_mean[-1]  # [T]
                            
                            data["attention_matrix"] = attn_mean
                            data["temporal_importance"] = temporal_imp

                        # ========================================
                        # ✅ Feature Importance (Horizon-Specific!)
                        # ========================================
                        if var_w is not None and config.compute_feature_importance:
                            var_sample = var_w[i].cpu().numpy()  # [T, F_after]
                            data["variable_importance_ts"] = var_sample
                            
                            # ✅ Horizon-specific feature importance
                            # Weighted by temporal importance
                            if temporal_imp is not None:
                                # feat_imp[f] = Σ_t (var_sample[t,f] * temporal_imp[t])
                                feat_imp = var_sample.T @ temporal_imp  # [F] = [F,T] @ [T]
                                feat_imp = feat_imp / (feat_imp.sum() + 1e-12)
                            else:
                                # Fallback: simple mean
                                feat_imp = var_sample.mean(axis=0)
                            
                            data["variable_importance"] = feat_imp

                            if model.has_news:
                                nb = model.num_basic_features
                                np_dim = model.news_projection_dim
                                basic_names = list(data_loader.feature_names[:nb])
                                news_names = [f"news_proj_{j}" for j in range(np_dim)]
                                feature_names_projected = basic_names + news_names
                                data["feature_names"] = np.array(feature_names_projected, dtype=object)
                                data["basic_importance"] = float(feat_imp[:nb].sum())
                                data["news_importance"]  = float(feat_imp[nb:].sum())
                            else:
                                data["feature_names"] = np.array(data_loader.feature_names, dtype=object)
                                data["basic_importance"] = float(feat_imp.sum())
                                data["news_importance"]  = 0.0

                        # NPZ 저장
                        out_npz = save_dir / f"{base_date}.npz"
                        np.savez_compressed(out_npz, **data)

        print("✓ Interpretation saved (no mean pooling)")

        import pandas as pd
        if pred_dir is not None and len(pred_rows) > 0:
            df_pred = pd.DataFrame(pred_rows)
            df_pred.to_csv(pred_dir / "val_predictions.csv", index=False)
            print(f"✓ Saved val predictions: {pred_dir / 'val_predictions.csv'}")

    print("\n✅ Training & interpretation completed.")


def main(config: TrainConfig):
    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = PREDICT_ROOT / path
        return str(path)

    config.data_dir = _resolve_path(config.data_dir)
    config.checkpoint_dir = _resolve_path(config.checkpoint_dir)
    config.output_dir = _resolve_path(config.output_dir)

    seq_lengths = [int(x) for x in getattr(config, "seq_lengths", []) or []]
    if seq_lengths:
        for seq_len in seq_lengths:
            cfg = replace(config, seq_length=seq_len)
            _run_single(cfg)
    else:
        _run_single(config)


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)
