from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tyro
from gluonts.dataset.common import ListDataset
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.model.deepar import DeepAREstimator
from lightning.pytorch.callbacks import EarlyStopping

MODEL_ROOT = Path(__file__).resolve().parents[1]
PREDICT_ROOT = MODEL_ROOT.parent
MODEL_SRC = MODEL_ROOT / "src"
for _path in (MODEL_SRC, PREDICT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_ROOT = PREDICT_ROOT.parent

from configs.deepar_config import DeepARConfig
from shared.data.bigquery_loader import load_news_features_bq, load_price_table
from data.dataset_deepar import build_multi_item_dataset, deepar_split
from shared.utils.set_seed import set_seed

# Suppress noisy runtime warnings in batch runs
warnings.filterwarnings("ignore")


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


def _resolve_price_bq_settings(cfg: DeepARConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("PRICE_BIGQUERY_PROJECT") or cfg.bq_project_id,
        "dataset_id": _get("PRICE_BIGQUERY_DATASET") or cfg.bq_dataset_id,
        "credentials_path": _get("PRICE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def _resolve_article_bq_settings(cfg: DeepARConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("ARTICLE_BIGQUERY_PROJECT"),
        "dataset_id": _get("ARTICLE_BIGQUERY_DATASET"),
        "credentials_path": _get("ARTICLE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "daily_table": _get("TABLE_DAILY_SUMMARY"),
    }


def _fetch_latest_inference_date(
    *,
    data_source: str,
    data_root: Path,
    project_id: Optional[str],
    dataset_id: Optional[str],
    table: Optional[str],
    credentials_path: Optional[str] = None,
) -> Optional[str]:
    if data_source == "bigquery":
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
        query = f"SELECT MAX(trade_date) AS max_date FROM `{project_id}.{dataset_id}.{table}`"
        try:
            rows = list(client.query(query).result())
        except Exception:
            return None
        if not rows:
            return None
        max_date = rows[0].get("max_date")
        if max_date is None:
            return None
        return str(max_date)[:10]

    infer_path = data_root / "inference_price.csv"
    if not infer_path.exists():
        return None
    try:
        df = pd.read_csv(infer_path)
    except Exception:
        return None
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return None
    max_dt = pd.to_datetime(df[date_col], errors="coerce").max()
    if pd.isna(max_dt):
        return None
    return max_dt.strftime("%Y-%m-%d")


def _load_price_df(
    *,
    data_source: str,
    data_dir: str,
    filename: str,
    commodity: str,
    bq_project_id: str,
    bq_dataset_id: str,
    bq_table: str,
) -> pd.DataFrame:
    if data_source == "bigquery":
        df = load_price_table(
            project_id=bq_project_id,
            dataset_id=bq_dataset_id,
            table=bq_table,
            commodity=commodity,
        )
    else:
        price_path = os.path.join(data_dir, filename)
        df = pd.read_csv(price_path)
        if "time" not in df.columns:
            raise ValueError(f"'time' column missing in {price_path}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["item_id"] = commodity
    return df


def _merge_news(price_df: pd.DataFrame, news_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return price_df
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    news = news_df.copy()
    if "date" not in news.columns and "collect_date" in news.columns:
        news["date"] = pd.to_datetime(news["collect_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = df.merge(news, on="date", how="left")
    if "news_count" in merged.columns:
        merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    news_emb_cols = [c for c in merged.columns if c.startswith("news_emb_")]
    if news_emb_cols:
        merged[news_emb_cols] = merged[news_emb_cols].fillna(0)
    return merged


def _build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    anchor = pd.Timestamp("2000-01-01")
    global_times = sorted(df["time"].unique())
    time2idx = {t: i for i, t in enumerate(global_times)}
    df = df.copy()
    df["time_idx_int"] = df["time"].map(time2idx)
    df["time_idx"] = anchor + pd.to_timedelta(df["time_idx_int"].astype(int), unit="D")
    return df


def _ensure_log_return1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df.columns:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    calc = np.log(df["close"] / df["close"].shift(1))
    if "log_return_1" in df.columns:
        df["log_return_1"] = df["log_return_1"].fillna(calc)
    else:
        df["log_return_1"] = calc
    return df


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["rv_5"] = df["log_return_1"].rolling(5).std()
    df["rv_10"] = df["log_return_1"].rolling(10).std()

    for col in ["open", "high", "EMA", "EMA_5", "EMA_10"]:
        if col in df.columns:
            df[col] = df[col] / df["close"]

    exclude_cols = {
        "time",
        "item_id",
        "close",
        "time_idx",
        "time_idx_int",
        "EMA_20",
        "EMA_50",
        "EMA_100",
        "Volume",
        "symbol",
        "trade_date",
        "date",
        "collect_date",
        "embedding",
        "news_embedding_mean",
        "key_word",
        "vol_return_7d",
        "vol_return_14d",
        "vol_return_21d",
    }
    # Drop timestamp-like or ingestion columns
    drop_keywords = ("created_at", "ingest_time")

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols or c.startswith("log_return_") or c.startswith("vol_return_"):
            continue
        if any(k in c for k in drop_keywords):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    # Avoid NaNs/Infs breaking StudentT loss
    numeric_cols = [c for c in (feature_cols + ["log_return_1"]) if c in df.columns]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df, feature_cols


def _fit_standard_scaler(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, np.ndarray]:
    feat_vals = df[feature_cols].to_numpy(dtype=float)
    feat_mean = np.nanmean(feat_vals, axis=0)
    feat_std = np.nanstd(feat_vals, axis=0)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)

    target_vals = df[target_col].to_numpy(dtype=float)
    target_mean = float(np.nanmean(target_vals))
    target_std = float(np.nanstd(target_vals))
    if target_std == 0 or np.isnan(target_std):
        target_std = 1.0

    return {
        "feature_names": np.array(feature_cols, dtype=object),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "target_mean": np.array([target_mean], dtype=float),
        "target_std": np.array([target_std], dtype=float),
    }


def _apply_standard_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_col: str,
    target_mean: float,
    target_std: float,
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = (out[feature_cols].to_numpy(dtype=float) - feature_mean) / feature_std
    if target_col in out.columns:
        out[target_col] = (out[target_col].to_numpy(dtype=float) - target_mean) / target_std
    return out


def _load_val_dates(split_file: str, fold_idx: int) -> List[str]:
    with open(split_file, "r") as f:
        data = json.load(f)
    val_dates = data["folds"][fold_idx]["val"]["t_dates"]
    return [str(pd.to_datetime(d).date()) for d in val_dates]


def _build_infer_dataset(df: pd.DataFrame, target_col: str, feature_cols: List[str], prediction_length: int):
    df = df.sort_values("time_idx").reset_index(drop=True)
    target = df[target_col].to_numpy(dtype=float)
    feats = np.vstack([df[c].to_numpy(dtype=float) for c in feature_cols])
    if prediction_length > 0:
        pad = np.repeat(feats[:, -1:], prediction_length, axis=1)
        feats = np.concatenate([feats, pad], axis=1)
    start = df["time_idx"].iloc[0]
    return ListDataset(
        [{"target": target, "start": start, "feat_dynamic_real": feats}],
        freq="D",
    )


def _eval_metrics_val_dates(
    df: pd.DataFrame,
    val_dates: List[str],
    predictor,
    *,
    target_col: str,
    feature_cols: List[str],
    context_length: int,
    prediction_length: int,
    horizons: List[int],
    num_samples: int,
    quantiles: List[float],
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    if not val_dates:
        return {"skipped": True, "reason": "no_val_dates"}

    df = df.sort_values("time_idx").reset_index(drop=True)
    df["date_str"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")

    q50 = 0.5 if 0.5 in quantiles else quantiles[len(quantiles) // 2]
    q10 = 0.1 if 0.1 in quantiles else min(quantiles)
    q90 = 0.9 if 0.9 in quantiles else max(quantiles)

    idx_map = {d: i for i, d in enumerate(df["date_str"].tolist())}
    val_indices = [idx_map[d] for d in val_dates if d in idx_map]
    if not val_indices:
        return {"skipped": True, "reason": "val_dates_not_in_df"}

    preds_by_h = {h: [] for h in horizons}
    trues_by_h = {h: [] for h in horizons}
    q10_by_h = {h: [] for h in horizons}
    q90_by_h = {h: [] for h in horizons}

    all_preds = []
    all_trues = []
    all_q10 = []
    all_q90 = []

    start = df["time_idx"].iloc[0]

    for t_idx in val_indices:
        base_idx = t_idx - 1  # h1 target is at t_idx
        if base_idx < context_length - 1:
            continue
        if base_idx + prediction_length >= len(df):
            continue

        target = df[target_col].iloc[: base_idx + 1].to_numpy(dtype=float)
        feats = np.vstack(
            [
                df[c]
                .iloc[: base_idx + 1 + prediction_length]
                .to_numpy(dtype=float)
                for c in feature_cols
            ]
        )
        dataset = ListDataset(
            [{"target": target, "start": start, "feat_dynamic_real": feats}],
            freq="D",
        )
        forecast = next(predictor.predict(dataset, num_samples=num_samples))
        true_future = df[target_col].iloc[base_idx + 1 : base_idx + 1 + prediction_length].to_numpy(dtype=float)
        if len(true_future) < prediction_length:
            continue

        pred_vals = forecast.quantile(q50)
        q10_vals = forecast.quantile(q10)
        q90_vals = forecast.quantile(q90)

        if target_mean is not None and target_std is not None:
            pred_vals = pred_vals * target_std + target_mean
            q10_vals = q10_vals * target_std + target_mean
            q90_vals = q90_vals * target_std + target_mean
            true_future = true_future * target_std + target_mean

        for h in horizons:
            idx = h - 1
            if idx >= len(true_future):
                continue
            preds_by_h[h].append(float(pred_vals[idx]))
            trues_by_h[h].append(float(true_future[idx]))
            q10_by_h[h].append(float(q10_vals[idx]))
            q90_by_h[h].append(float(q90_vals[idx]))

        all_preds.extend(pred_vals.tolist())
        all_trues.extend(true_future.tolist())
        all_q10.extend(q10_vals.tolist())
        all_q90.extend(q90_vals.tolist())

    if not all_preds:
        return {"skipped": True, "reason": "no_eval_windows"}

    all_preds = np.array(all_preds, dtype=float)
    all_trues = np.array(all_trues, dtype=float)
    all_q10 = np.array(all_q10, dtype=float)
    all_q90 = np.array(all_q90, dtype=float)

    def _metrics(preds: np.ndarray, trues: np.ndarray, q10_vals=None, q90_vals=None):
        mae = float(np.mean(np.abs(preds - trues)))
        rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
        ss_res = float(np.sum((trues - preds) ** 2))
        ss_tot = float(np.sum((trues - np.mean(trues)) ** 2))
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        da = float(np.mean((preds > 0) == (trues > 0)) * 100.0)
        out = {"MAE": mae, "RMSE": rmse, "R2": r2, "DA": da}
        if q10_vals is not None and q90_vals is not None:
            coverage = float(np.mean((trues >= q10_vals) & (trues <= q90_vals)) * 100.0)
            out["coverage_q10_q90"] = coverage
        return out

    per_h = {}
    for h in horizons:
        if not preds_by_h[h]:
            continue
        pred_arr = np.array(preds_by_h[h], dtype=float)
        true_arr = np.array(trues_by_h[h], dtype=float)
        q10_arr = np.array(q10_by_h[h], dtype=float)
        q90_arr = np.array(q90_by_h[h], dtype=float)
        per_h[str(h)] = _metrics(pred_arr, true_arr, q10_arr, q90_arr)

    overall = _metrics(all_preds, all_trues, all_q10, all_q90)
    return {"overall": overall, "per_horizon": per_h}


def main(cfg: DeepARConfig) -> None:
    set_seed(cfg.seed)

    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = PREDICT_ROOT / path
        return str(path)

    cfg.data_dir = _resolve_path(cfg.data_dir)
    cfg.split_file = _resolve_path(cfg.split_file)
    cfg.future_price_file = _resolve_path(cfg.future_price_file)
    cfg.checkpoint_root = _resolve_path(cfg.checkpoint_root)
    cfg.prediction_root = _resolve_path(cfg.prediction_root)
    cfg.combined_output = _resolve_path(cfg.combined_output)

    price_bq = _resolve_price_bq_settings(cfg)
    if price_bq.get("credentials_path") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = price_bq["credentials_path"]
    if price_bq.get("project_id"):
        cfg.bq_project_id = str(price_bq["project_id"])
    if price_bq.get("dataset_id"):
        cfg.bq_dataset_id = str(price_bq["dataset_id"])

    article_bq = _resolve_article_bq_settings(cfg)

    data_root = Path(cfg.data_dir).resolve()

    train_raw = _load_price_df(
        data_source=cfg.data_source,
        data_dir=cfg.data_dir,
        filename="train_price.csv",
        commodity=cfg.target_commodity,
        bq_project_id=cfg.bq_project_id,
        bq_dataset_id=cfg.bq_dataset_id,
        bq_table=cfg.bq_train_table,
    )
    infer_raw = None
    if cfg.data_source == "bigquery":
        try:
            infer_raw = _load_price_df(
                data_source=cfg.data_source,
                data_dir=cfg.data_dir,
                filename="inference_price.csv",
                commodity=cfg.target_commodity,
                bq_project_id=cfg.bq_project_id,
                bq_dataset_id=cfg.bq_dataset_id,
                bq_table=cfg.bq_inference_table,
            )
        except Exception:
            infer_raw = None
    else:
        infer_path = os.path.join(cfg.data_dir, "inference_price.csv")
        if os.path.exists(infer_path):
            infer_raw = _load_price_df(
                data_source=cfg.data_source,
                data_dir=cfg.data_dir,
                filename="inference_price.csv",
                commodity=cfg.target_commodity,
                bq_project_id=cfg.bq_project_id,
                bq_dataset_id=cfg.bq_dataset_id,
                bq_table=cfg.bq_inference_table,
            )

    news_df = None
    if article_bq.get("project_id") and article_bq.get("dataset_id"):
        news_df = load_news_features_bq(
            project_id=article_bq.get("project_id"),
            dataset_id=article_bq.get("dataset_id"),
            table=article_bq.get("daily_table") or "daily_summary",
            commodity=cfg.target_commodity,
            credentials_path=article_bq.get("credentials_path"),
        )

    if infer_raw is not None:
        df = pd.concat([train_raw, infer_raw], axis=0, ignore_index=True)
        if df["time"].duplicated().any():
            dup_dates = (
                df.loc[df["time"].duplicated(keep=False), "time"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            sample = ", ".join(dup_dates[:10])
            raise ValueError(
                "Duplicate dates across train/inference detected. "
                f"Sample duplicates: {sample}. "
                "Please ensure train_price and inference_price do not overlap."
            )
        df = df.sort_values("time")
    else:
        df = train_raw.copy()

    df = _merge_news(df, news_df)
    df = _ensure_log_return1(df)
    df, feature_cols = _prepare_features(df)
    df = _build_time_index(df)

    # split (train/val) from json
    split_file = cfg.split_file
    if "{commodity}" in split_file:
        split_file = split_file.format(commodity=cfg.target_commodity)
    split_path = Path(split_file)
    if not split_path.is_absolute():
        candidate = PREDICT_ROOT / split_path
        if candidate.exists():
            split_path = candidate
        else:
            split_path = (data_root / split_path).resolve()
    else:
        split_path = split_path.resolve()

    # determine output date from latest inference price
    latest_infer_date = _fetch_latest_inference_date(
        data_source=cfg.data_source,
        data_root=data_root,
        project_id=cfg.bq_project_id,
        dataset_id=cfg.bq_dataset_id,
        table=cfg.bq_inference_table,
        credentials_path=price_bq.get("credentials_path"),
    )
    if latest_infer_date:
        as_of = latest_infer_date
    elif infer_raw is not None and not infer_raw.empty:
        as_of = str(pd.to_datetime(infer_raw["time"]).max())[:10]
    else:
        as_of = str(pd.to_datetime(train_raw["time"]).max())[:10]

    tag = f"_{cfg.output_tag}" if cfg.output_tag else ""
    ckpt_root = Path(cfg.checkpoint_root.format(commodity=cfg.target_commodity, date=as_of, tag=tag))
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for fold in cfg.fold:
        train_df, val_df = deepar_split(df, str(split_path), fold)
        val_dates = _load_val_dates(str(split_path), fold)

        scaler = _fit_standard_scaler(train_df, feature_cols, "log_return_1")
        feature_mean = scaler["feature_mean"]
        feature_std = scaler["feature_std"]
        target_mean = float(scaler["target_mean"][0])
        target_std = float(scaler["target_std"][0])

        train_df = _apply_standard_scaler(
            train_df,
            feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_col="log_return_1",
            target_mean=target_mean,
            target_std=target_std,
        )
        val_df = _apply_standard_scaler(
            val_df,
            feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_col="log_return_1",
            target_mean=target_mean,
            target_std=target_std,
        )

        scaler_path = ckpt_root / "scaler.npz"
        np.savez(scaler_path, **scaler)

        for ctx_len in cfg.seq_lengths:
            print(f"\n▶ DeepAR TRAIN window={ctx_len} fold={fold}")
            w_ckpt = ckpt_root / f"w{ctx_len}"
            w_ckpt.mkdir(parents=True, exist_ok=True)

            train_ds = build_multi_item_dataset({cfg.target_commodity: train_df}, "log_return_1", feature_cols)
            val_ds = build_multi_item_dataset({cfg.target_commodity: val_df}, "log_return_1", feature_cols)

            callbacks = None
            if cfg.early_stop:
                callbacks = [
                    EarlyStopping(
                        monitor="val_loss",
                        mode="min",
                        patience=cfg.patience,
                        min_delta=cfg.min_delta,
                    )
                ]

            estimator = DeepAREstimator(
                freq="D",
                prediction_length=cfg.prediction_length,
                context_length=ctx_len,
                num_feat_dynamic_real=len(feature_cols),
                num_layers=3,
                hidden_size=64,
                dropout_rate=0.1,
                lr=1e-4,
                scaling=False,
                distr_output=StudentTOutput(),
                trainer_kwargs={
                    "max_epochs": cfg.epochs,
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "gradient_clip_val": 1.0,
                    "callbacks": callbacks,
                },
            )

            predictor = estimator.train(training_data=train_ds, validation_data=val_ds)
            model_dir = w_ckpt / "best_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            predictor.serialize(model_dir)

            # validation metrics (log_return-based)
            df_for_val = df[df["time"].isin(train_raw["time"])].reset_index(drop=True)
            df_for_val = _apply_standard_scaler(
                df_for_val,
                feature_cols,
                feature_mean=feature_mean,
                feature_std=feature_std,
                target_col="log_return_1",
                target_mean=target_mean,
                target_std=target_std,
            )
            val_metrics = _eval_metrics_val_dates(
                df_for_val,
                val_dates,
                predictor,
                target_col="log_return_1",
                feature_cols=feature_cols,
                context_length=ctx_len,
                prediction_length=cfg.prediction_length,
                horizons=cfg.horizons,
                num_samples=cfg.num_samples,
                quantiles=cfg.quantiles,
                target_mean=target_mean,
                target_std=target_std,
            )
            metrics_path = w_ckpt / "val_metrics.json"
            metrics_payload = {
                "commodity": cfg.target_commodity,
                "fold": fold,
                "window_size": ctx_len,
                "horizons": cfg.horizons,
                "metrics": val_metrics,
            }
            metrics_path.write_text(json.dumps(metrics_payload, indent=2))
            print(f"✓ Saved checkpoint: {model_dir}")
            print(f"✓ Saved val metrics: {metrics_path}")


if __name__ == "__main__":
    main(tyro.cli(DeepARConfig))
