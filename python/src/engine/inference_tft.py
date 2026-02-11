from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import BDay

from src.data.dataset_tft import TFTDataLoader
from src.data.bigquery_loader import load_price_table, load_news_features_bq
from src.models.TFT import TemporalFusionTransformer
from src.utils.unified_output import build_forecast_payload, map_horizon_values, write_json


def _load_checkpoint(checkpoint_path: Path, device: str) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def _resolve_checkpoint_path(
    commodity: str,
    fold: int,
    horizons: Sequence[int],
    checkpoint_path: Optional[str],
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    h_tag = "h" + "-".join(map(str, horizons))
    return (
        Path("src/outputs/checkpoints")
        / f"TFT_{commodity}_fold{fold}_{h_tag}"
        / "best_model.pt"
    )


def _load_scaler(checkpoint_path: Path) -> Optional[Dict[str, np.ndarray]]:
    scaler_path = checkpoint_path.parent / "scaler.npz"
    if not scaler_path.exists():
        return None
    sc = np.load(scaler_path, allow_pickle=True)
    return {
        "scale_x": bool(sc.get("scale_x", False)),
        "scale_y": bool(sc.get("scale_y", False)),
        "x_mean": sc.get("x_mean", None),
        "x_std": sc.get("x_std", None),
        "y_mean": sc.get("y_mean", None),
        "y_std": sc.get("y_std", None),
        "feature_names": sc.get("feature_names", None),
    }


def _resolve_output_root(
    commodity: str,
    exp_name: str,
    fold: int,
    split: str,
    output_root: Optional[str],
) -> Path:
    if output_root is not None:
        return Path(output_root)
    return (
        Path("src/outputs/predictions/unified/tft")
        / commodity
        / exp_name
        / f"fold_{fold}"
        / split
        / "results"
    )


def _normalize_dates(dates: Sequence) -> List[str]:
    return [str(d)[:10] for d in dates]

def _ensure_unique_dates(df: pd.DataFrame, date_col: str, label: str) -> None:
    if date_col not in df.columns:
        return
    date_series = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    dup_mask = date_series.duplicated()
    if dup_mask.any():
        dup_dates = date_series[dup_mask].dropna().unique().tolist()
        sample = ", ".join(dup_dates[:5])
        raise ValueError(f"Duplicate dates in {label}: {sample}")


def _build_trading_calendar(data_dir: Path, commodity: str) -> List[str]:
    candidates = [
        data_dir / "test_price.csv",
        data_dir / "inference_price.csv",
        data_dir / "train_price.csv",
        data_dir.parent.parent / f"{commodity}_future_price.csv",
    ]
    dates: List[str] = []
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        date_col = None
        for cand in ("time", "trade_date", "date"):
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            continue
        dates.extend(
            pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").tolist()
        )
    dates = [d for d in dates if d and d != "NaT"]
    return sorted(set(dates))


def _resolve_exchange_for_commodity(commodity: Optional[str]) -> str:
    if not commodity:
        return "NYSE"
    key = str(commodity).strip()
    mapping = {
        "corn": "CBOT",
        "wheat": "CBOT",
        "soybean": "CBOT",
        "gold": "COMEX",
        "silver": "COMEX",
        "copper": "COMEX",
        "ZC=F": "CBOT",
        "ZW=F": "CBOT",
        "ZS=F": "CBOT",
        "GC=F": "COMEX",
        "SI=F": "COMEX",
        "HG=F": "COMEX",
    }
    if key in mapping:
        return mapping[key]
    lower = key.lower()
    if lower in mapping:
        return mapping[lower]
    return "NYSE"


def _exchange_future_dates(
    as_of: str,
    horizons: Sequence[int],
    exchange: str = "NYSE",
) -> Dict[int, str]:
    if not horizons:
        return {}
    try:
        import pandas_market_calendars as mcal
    except Exception:
        return {}
    try:
        calendar = mcal.get_calendar(exchange)
    except Exception:
        return {}

    base = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(base):
        return {}
    max_h = max(int(h) for h in horizons)
    if max_h <= 0:
        return {}
    start = base + pd.Timedelta(days=1)
    end = base + pd.Timedelta(days=max_h * 5 + 7)
    try:
        valid_days = calendar.valid_days(start_date=start, end_date=end)
    except Exception:
        return {}
    day_list = [d.date().isoformat() for d in valid_days]
    mapping: Dict[int, str] = {}
    for h in horizons:
        idx = int(h) - 1
        if 0 <= idx < len(day_list):
            mapping[int(h)] = day_list[idx]
    return mapping


def _next_trading_dates(
    as_of: str,
    horizons: Sequence[int],
    trading_calendar: Sequence[str],
    exchange: str = "NYSE",
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if trading_calendar and as_of in trading_calendar:
        idx = trading_calendar.index(as_of)
        future = trading_calendar[idx + 1 :]
        for h in horizons:
            if h - 1 < len(future):
                mapping[int(h)] = future[h - 1]
    missing = [int(h) for h in horizons if int(h) not in mapping]
    if missing:
        mapping.update(_exchange_future_dates(as_of, missing, exchange=exchange))
    base = pd.to_datetime(as_of)
    for h in horizons:
        if int(h) not in mapping:
            mapping[int(h)] = (base + BDay(int(h))).strftime("%Y-%m-%d")
    return mapping


def _build_window_date_lookup(data: pd.DataFrame) -> Dict[str, int]:
    dates = pd.to_datetime(data["time"]).dt.strftime("%Y-%m-%d").tolist()
    return {d: i for i, d in enumerate(dates)}


def _window_dates_for_target(
    date_str: str,
    date_to_idx: Dict[str, int],
    all_dates: List[str],
    seq_length: int,
) -> Optional[List[str]]:
    idx = date_to_idx.get(date_str)
    if idx is None:
        return None
    start = idx - seq_length
    if start < 0:
        return None
    return all_dates[start:idx]


def _prepare_horizon_groups(horizons: Sequence[int], groups: Sequence[int]) -> Dict[int, List[int]]:
    group_indices: Dict[int, List[int]] = {}
    for g in groups:
        idxs = [i for i, h in enumerate(horizons) if h <= g]
        if idxs:
            group_indices[int(g)] = idxs
    return group_indices


def _save_importance_files(
    *,
    out_dir: Path,
    date_str: str,
    window_dates: List[str],
    feature_names: List[str],
    var_weights: np.ndarray,  # [T, F]
    horizon_attn: np.ndarray,  # [H, T]
    horizon_groups: Dict[int, List[int]],
    top_k: int,
    save_images: bool,
) -> None:
    if var_weights.ndim != 2:
        return

    if var_weights.shape[1] != len(feature_names):
        min_f = min(var_weights.shape[1], len(feature_names))
        feature_names = feature_names[:min_f]
        var_weights = var_weights[:, :min_f]

    if var_weights.shape[0] != horizon_attn.shape[1]:
        # shape mismatch between temporal weights and feature weights
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    flatten_groups = len(horizon_groups) == 1
    for group, idxs in horizon_groups.items():
        if flatten_groups:
            group_dir = out_dir / date_str
        else:
            group_dir = out_dir / f"h{group}" / date_str
        group_dir.mkdir(parents=True, exist_ok=True)

        temporal_weights = horizon_attn[idxs].mean(axis=0)
        denom = float(temporal_weights.sum()) + 1e-12
        temporal_weights = temporal_weights / denom

        # Temporal importance (per timestep)
        df_temporal = pd.DataFrame(
            {"timestep": window_dates, "temporal_importance": temporal_weights}
        )
        df_temporal.to_csv(group_dir / "temporal_importance.csv", index=False)

        # Feature importance (weighted by temporal weights)
        feat_importance = (var_weights * temporal_weights[:, None]).sum(axis=0)
        total = float(feat_importance.sum()) + 1e-12
        df_feat = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": feat_importance,
                "importance_percent": feat_importance / total * 100.0,
            }
        ).sort_values("importance", ascending=False)
        df_feat.to_csv(group_dir / "feature_importance_full.csv", index=False)

        # Group importance (Basic vs News)
        news_mask = np.array(["news" in f.lower() for f in feature_names], dtype=bool)
        news_imp = float(feat_importance[news_mask].sum()) if news_mask.any() else 0.0
        basic_imp = float(feat_importance.sum()) - news_imp
        df_group = pd.DataFrame(
            [
                {
                    "group": "Basic(Price+Tech)",
                    "importance": basic_imp,
                    "ratio_percent": basic_imp / total * 100.0,
                },
                {
                    "group": "News",
                    "importance": news_imp,
                    "ratio_percent": news_imp / total * 100.0,
                },
            ]
        )
        df_group.to_csv(group_dir / "group_importance.csv", index=False)

        # Temporal feature importance (Top-K features)
        top_features = df_feat.head(top_k)["feature"].tolist()
        top_indices = [feature_names.index(f) for f in top_features]
        temporal_feature = var_weights * temporal_weights[:, None]
        df_tf = pd.DataFrame(
            temporal_feature[:, top_indices].T,
            index=top_features,
            columns=window_dates,
        )
        df_tf.to_csv(group_dir / "temporal_feature_importance_topk.csv")

        if save_images:
            # Feature importance bar
            plt.figure(figsize=(10, 6))
            plt.barh(
                range(len(top_features)),
                df_feat.head(top_k)["importance"].values,
                color="steelblue",
            )
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel("Importance")
            plt.title(f"Feature Importance Top-{top_k} (h<= {group})")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(group_dir / "feature_importance.png", dpi=200)
            plt.close()

            # Temporal importance bar
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(window_dates)), temporal_weights, color="orange")
            plt.xticks(
                range(len(window_dates)),
                window_dates,
                rotation=45,
                fontsize=8,
            )
            plt.title(f"Temporal Importance (h<= {group})")
            plt.tight_layout()
            plt.savefig(group_dir / "temporal_importance.png", dpi=200)
            plt.close()

            # Temporal feature importance heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                df_tf.values,
                xticklabels=window_dates,
                yticklabels=top_features,
                cmap="YlOrRd",
                cbar_kws={"label": "Importance"},
            )
            plt.xlabel("Date")
            plt.ylabel("Feature")
            plt.title(f"Temporal Feature Importance (Top-{top_k}, h<= {group})")
            plt.tight_layout()
            plt.savefig(group_dir / "temporal_feature_importance.png", dpi=200)
            plt.close()


def run_inference_tft(
    commodity: str,
    fold: int,
    seq_length: int,
    horizons: Sequence[int],
    *,
    exp_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_root: Optional[str] = None,
    data_dir: str = "src/datasets",
    split_file: Optional[str] = None,
    data_source: str = "local",
    bq_project_id: Optional[str] = None,
    bq_dataset_id: Optional[str] = None,
    bq_train_table: str = "train_price",
    bq_inference_table: str = "inference_price",
    news_source: str = "csv",
    bq_news_project_id: Optional[str] = None,
    bq_news_dataset_id: Optional[str] = None,
    bq_news_table: str = "daily_summary",
    split: str = "test",
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    use_variable_selection: bool = True,
    quantiles: Optional[Sequence[float]] = None,
    include_targets: bool = True,
    scale_x: bool = True,
    scale_y: bool = True,
    save_importance: bool = False,
    importance_groups: Optional[Sequence[int]] = None,
    importance_top_k: int = 20,
    save_importance_images: bool = False,
    save_prediction_plot: bool = False,
    interpretations_use_fold_dir: bool = True,
) -> Path:
    """
    Run TFT inference and write per-date unified JSON outputs.
    """
    if split not in {"test", "val", "inference"}:
        raise ValueError("split must be 'test', 'val', or 'inference'.")

    if split == "inference":
        include_targets = False

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = exp_name or f"tft_w{seq_length}_h{'-'.join(map(str, horizons))}"
    ckpt_path = _resolve_checkpoint_path(commodity, fold, horizons, checkpoint_path)
    # Load scaler early so we can enforce the training feature order.
    scaler = _load_scaler(ckpt_path)
    feature_columns = None
    if scaler is not None:
        feature_names = scaler.get("feature_names", None)
        if feature_names is not None:
            # np.load returns array-like; convert to list of strings.
            feature_columns = [str(f) for f in feature_names.tolist()]

    price_file = f"preprocessing/{commodity}_feature_engineering.csv"
    news_file = "news_features.csv"

    price_source = str(Path(data_dir) / price_file)
    window_len = seq_length + 1
    if data_source == "bigquery":
        if not bq_project_id or not bq_dataset_id:
            raise ValueError("BigQuery project_id and dataset_id are required.")
        if split == "inference":
            infer_df = load_price_table(
                project_id=bq_project_id,
                dataset_id=bq_dataset_id,
                table=bq_inference_table,
                commodity=commodity,
            )
            infer_df = infer_df.sort_values("time").reset_index(drop=True)
            if len(infer_df) < window_len:
                pad_len = window_len - len(infer_df)
                train_df = load_price_table(
                    project_id=bq_project_id,
                    dataset_id=bq_dataset_id,
                    table=bq_train_table,
                    commodity=commodity,
                )
                infer_start = infer_df["time"].min()
                train_tail = (
                    train_df[train_df["time"] < infer_start]
                    .sort_values("time")
                    .tail(pad_len)
                )
                price_source = pd.concat([train_tail, infer_df], ignore_index=True)
            else:
                price_source = infer_df.tail(window_len)
            _ensure_unique_dates(price_source, "time", "inference_price (bq merged)")
            price_source = price_source.sort_values("time").reset_index(drop=True)
            if len(price_source) < window_len:
                raise ValueError(
                    f"Not enough rows for inference window: need {window_len}, got {len(price_source)}"
                )
        else:
            price_source = load_price_table(
                project_id=bq_project_id,
                dataset_id=bq_dataset_id,
                table=bq_train_table,
                commodity=commodity,
            )
            _ensure_unique_dates(price_source, "time", "train_price (bq)")
    elif split == "inference":
        infer_path = Path(data_dir) / "inference_price.csv"
        if infer_path.exists():
            infer_df = pd.read_csv(infer_path)
        else:
            infer_df = pd.read_csv(price_source)
        _ensure_unique_dates(infer_df, "time", "inference_price")
        infer_df = infer_df.sort_values("time").reset_index(drop=True)
        if len(infer_df) < window_len:
            pad_len = window_len - len(infer_df)
            train_path = Path(data_dir) / "train_price.csv"
            if train_path.exists():
                train_df = pd.read_csv(train_path)
            else:
                train_df = pd.read_csv(price_source)
            _ensure_unique_dates(train_df, "time", "train_price")
            infer_start = infer_df["time"].min()
            train_tail = (
                train_df[train_df["time"] < infer_start]
                .sort_values("time")
                .tail(pad_len)
            )
            price_source = pd.concat([train_tail, infer_df], ignore_index=True)
        else:
            price_source = infer_df.tail(window_len)
        _ensure_unique_dates(price_source, "time", "inference_price (merged)")
        price_source = price_source.sort_values("time").reset_index(drop=True)
        if len(price_source) < window_len:
            raise ValueError(
                f"Not enough rows for inference window: need {window_len}, got {len(price_source)}"
            )

    split_path = Path(split_file) if split_file else Path(data_dir) / "rolling_fold.json"
    if not split_path.is_absolute():
        candidate = Path(data_dir) / split_path
        if candidate.exists():
            split_path = candidate

    news_data_source = str(Path(data_dir) / news_file)
    if news_source == "bigquery":
        if not bq_news_project_id or not bq_news_dataset_id:
            raise ValueError("BigQuery news project_id and dataset_id are required.")
        news_data_source = load_news_features_bq(
            project_id=bq_news_project_id,
            dataset_id=bq_news_dataset_id,
            table=bq_news_table,
            commodity=commodity,
        )

    data_loader = TFTDataLoader(
        price_data_path=price_source,
        news_data_path=news_data_source,
        split_file=str(split_path),
        seq_length=seq_length,
        horizons=list(horizons),
        batch_size=batch_size,
        num_workers=num_workers,
        feature_columns=feature_columns,
        require_targets=(split != "inference"),
        seed=seed,
    )

    # Load scaler saved during training so inference matches training preprocessing.
    y_mean = None
    y_std = None
    if scaler is not None:
        data_loader.fold_scalers[fold] = scaler
        if scaler.get("scale_y", False):
            y_mean = scaler.get("y_mean")
            y_std = scaler.get("y_std")

    if split == "test":
        dates, data_loader_iter = data_loader.get_test_loader(
            fold, scale_x=scale_x, scale_y=scale_y
        )
    elif split == "inference":
        dates, data_loader_iter = data_loader.get_inference_loader(
            scale_x=scale_x, scale_y=scale_y
        )
    else:
        _, data_loader_iter, valid_dates = data_loader.get_fold_loaders(
            fold, scale_x=scale_x, scale_y=scale_y
        )
        dates = valid_dates

    dates = _normalize_dates(dates)
    trading_calendar = _build_trading_calendar(Path(data_dir), commodity)
    all_dates = pd.to_datetime(data_loader.data["time"]).dt.strftime("%Y-%m-%d").tolist()
    date_to_idx = _build_window_date_lookup(data_loader.data)
    date_to_close: Dict[str, float] = {}
    if "close" in data_loader.data.columns:
        close_series = (
            data_loader.data[["time", "close"]]
            .dropna()
            .assign(date=lambda df: pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d"))
        )
        date_to_close = dict(zip(close_series["date"].tolist(), close_series["close"].tolist()))
    horizon_groups = _prepare_horizon_groups(horizons, importance_groups or [])

    checkpoint = _load_checkpoint(ckpt_path, device)
    model_cfg = checkpoint.get("config", {})

    model = TemporalFusionTransformer(
        num_features=data_loader.X.shape[-1],
        num_horizons=len(horizons),
        hidden_dim=model_cfg.get("hidden_dim", 32),
        lstm_layers=model_cfg.get("num_layers", 2),
        attention_heads=model_cfg.get("attention_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        use_variable_selection=use_variable_selection,
        quantiles=list(quantiles) if quantiles is not None else None,
        news_projection_dim=model_cfg.get("news_projection_dim", 32),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Use projected feature names for interpretation (match training behavior)
    if model.has_news:
        nb = model.num_basic_features
        np_dim = model.news_projection_dim
        basic_names = list(data_loader.feature_names[:nb])
        news_names = [f"news_proj_{j}" for j in range(np_dim)]
        importance_feature_names = basic_names + news_names
    else:
        importance_feature_names = list(data_loader.feature_names)

    all_preds = []
    all_trues = []
    offset = 0

    if quantiles:
        if 0.5 in quantiles:
            median_idx = list(quantiles).index(0.5)
        else:
            median_idx = len(quantiles) // 2
    else:
        median_idx = None

    with torch.no_grad():
        for x_batch, y_batch in data_loader_iter:
            x_batch = x_batch.to(device)
            outputs = model(x_batch, return_attention=save_importance)
            preds = outputs["predictions"].detach().cpu().numpy()
            all_preds.append(preds)
            all_trues.append(y_batch.numpy())

            if save_importance:
                var_weights = outputs.get("variable_importance")
                horizon_attn = outputs.get("horizon_attention_weights")
                if var_weights is None or horizon_attn is None:
                    offset += len(y_batch)
                    continue

                var_weights = var_weights.detach().cpu().numpy()  # [B,T,F]
                horizon_attn = horizon_attn.detach().cpu().numpy()  # [B,heads,H,T]

                batch_dates = dates[offset : offset + len(y_batch)]
                for i, date_str in enumerate(batch_dates):
                    window_dates = _window_dates_for_target(
                        date_str, date_to_idx, all_dates, seq_length
                    )
                    if window_dates is None or len(window_dates) != seq_length:
                        continue

                    out_dir = (
                        _resolve_output_root(commodity, exp_name, fold, split, output_root)
                        .parent
                        / "interpretations"
                    )
                    if interpretations_use_fold_dir:
                        out_dir = out_dir / f"fold_{fold}"

                    attn = horizon_attn[i].mean(axis=0)  # [H,T]

                    _save_importance_files(
                        out_dir=out_dir,
                        date_str=date_str,
                        window_dates=window_dates,
                        feature_names=importance_feature_names,
                        var_weights=var_weights[i],
                        horizon_attn=attn,
                        horizon_groups=horizon_groups,
                        top_k=importance_top_k,
                        save_images=save_importance_images,
                    )

            if save_prediction_plot and date_to_close:
                batch_dates = dates[offset : offset + len(y_batch)]
                for i, date_str in enumerate(batch_dates):
                    window_dates = _window_dates_for_target(
                        date_str, date_to_idx, all_dates, seq_length
                    )
                    if window_dates is None:
                        continue

                    history_dates = list(window_dates)
                    history_close = [date_to_close.get(d) for d in history_dates]
                    if any(v is None for v in history_close):
                        continue

                    base_close = date_to_close.get(date_str)
                    if base_close is None or not np.isfinite(base_close):
                        continue
                    if not history_dates or history_dates[-1] != date_str:
                        history_dates.append(date_str)
                        history_close.append(base_close)

                    if preds.ndim == 3 and median_idx is not None:
                        pred_row = preds[i, :, median_idx]
                    else:
                        pred_row = preds[i]

                    if y_mean is not None and y_std is not None:
                        pred_row = pred_row * y_std + y_mean

                    pred_date_map = _next_trading_dates(
                        date_str,
                        horizons,
                        trading_calendar,
                        exchange=_resolve_exchange_for_commodity(commodity),
                    )
                    pred_dates = [pred_date_map[int(h)] for h in horizons]
                    pred_close = [float(base_close * np.exp(v)) for v in pred_row]

                    plots_dir = (
                        _resolve_output_root(commodity, exp_name, fold, split, output_root)
                        .parent
                        / "plots"
                    )
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    plt.figure(figsize=(10, 4))
                    plt.plot(history_dates, history_close, label="History Close", color="black")
                    plt.axvline(date_str, color="gray", linestyle="--", linewidth=1)
                    plt.plot(pred_dates, pred_close, label="Pred Close", color="tab:blue", marker="o")
                    plt.xticks(rotation=45, fontsize=8)
                    plt.title(f"{commodity.upper()} Close Forecast (as_of {date_str})")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{date_str}_forecast.png", dpi=200)
                    plt.close()

            offset += len(y_batch)

    if not all_preds:
        return _resolve_output_root(commodity, exp_name, fold, split, output_root)

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    # Inverse scale predictions/targets if y-scaling was used during training.
    if y_mean is not None and y_std is not None:
        if preds.ndim == 3:
            preds = preds * y_std[None, :, None] + y_mean[None, :, None]
        else:
            preds = preds * y_std[None, :] + y_mean[None, :]
        if trues is not None and trues.size:
            trues = trues * y_std[None, :] + y_mean[None, :]

    if len(dates) != preds.shape[0]:
        raise ValueError(f"Date count {len(dates)} does not match predictions {preds.shape[0]}.")

    output_root_path = _resolve_output_root(commodity, exp_name, fold, split, output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    if preds.ndim == 3:
        if quantiles and 0.5 in quantiles:
            median_idx = list(quantiles).index(0.5)
        else:
            median_idx = preds.shape[2] // 2
    else:
        median_idx = None

    for idx, date_str in enumerate(dates):
        if preds.ndim == 3:
            point_pred = preds[idx, :, median_idx]
            quantile_payload = None
            if quantiles and len(quantiles) == preds.shape[2]:
                quantile_payload = {
                    f"q{q:.2f}": preds[idx, :, q_idx]
                    for q_idx, q in enumerate(quantiles)
                }
        else:
            point_pred = preds[idx]
            quantile_payload = None

        target_vals = trues[idx] if include_targets else None

        pred_date_map = _next_trading_dates(
            date_str,
            horizons,
            trading_calendar,
            exchange=_resolve_exchange_for_commodity(commodity),
        )
        pred_dates_payload = {f"h{int(h)}": pred_date_map[int(h)] for h in horizons}

        payload = build_forecast_payload(
            model="tft",
            commodity=commodity,
            window=seq_length,
            horizons=horizons,
            as_of=date_str,
            fold=fold,
            predictions=point_pred,
            targets=target_vals,
            quantiles=quantile_payload,
            model_variant=exp_name,
            extra_meta={
                "split": split,
            },
            prediction_dates=pred_dates_payload,
        )

        base_close = date_to_close.get(date_str)
        if base_close is not None and np.isfinite(base_close):
            close_pred = [float(base_close * np.exp(v)) for v in point_pred]
            payload["predictions"]["close"] = map_horizon_values(close_pred, horizons)

        output_path = output_root_path / f"{date_str}.json"
        write_json(payload, output_path)

    return output_root_path
