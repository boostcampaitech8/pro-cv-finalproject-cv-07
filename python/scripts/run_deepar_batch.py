"""
DeepAR deployment runner (train + inference) for window sizes 5/20/60.
Keeps model structure identical to existing DeepAR settings (restart.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import List, Dict
import json

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import torch
import tyro

from lightning.pytorch.callbacks import EarlyStopping
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions

from src.configs.deepar_config import DeepARConfig
from src.data.dataset import build_multi_item_dataset, deepar_split
from src.data.bigquery_loader import load_price_table
from src.utils.set_seed import set_seed


def _resolve_future_price_path(data_dir: str, commodity: str, future_price_file: str) -> Path | None:
    candidates = [
        Path(future_price_file.format(commodity=commodity)),
        Path(data_dir) / "future_price.csv",
        Path(data_dir).parents[1] / f"{commodity}_future_price.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_test_price_path(data_dir: str, commodity: str) -> Path | None:
    candidates = [
        Path(data_dir) / "test_price.csv",
        Path(data_dir) / f"{commodity}_test_price.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_future_close(
    data_dir: str,
    commodity: str,
    as_of: str,
    prediction_length: int,
    future_price_file: str,
) -> tuple[list[str], np.ndarray] | None:
    future_path = _resolve_future_price_path(data_dir, commodity, future_price_file)
    if future_path is None:
        return None
    future_df = pd.read_csv(future_path)
    if "time" not in future_df.columns or "close" not in future_df.columns:
        return None
    future_df["time"] = pd.to_datetime(future_df["time"], errors="coerce")
    future_df = future_df.dropna(subset=["time"]).sort_values("time")
    future_df = future_df[future_df["time"] > pd.to_datetime(as_of)]
    if future_df.empty:
        return None
    future_df = future_df.head(prediction_length)
    if len(future_df) == 0:
        return None
    dates = future_df["time"].dt.strftime("%Y-%m-%d").tolist()
    closes = future_df["close"].to_numpy(dtype=float)
    return dates, closes


def _load_future_log_return(
    data_dir: str,
    commodity: str,
    as_of: str,
    prediction_length: int,
) -> tuple[list[str], np.ndarray] | None:
    test_path = _resolve_test_price_path(data_dir, commodity)
    if test_path is None:
        return None
    df = pd.read_csv(test_path)
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None or "log_return_1" not in df.columns:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df[df[date_col] > pd.to_datetime(as_of)]
    if df.empty:
        return None
    df = df.head(prediction_length)
    if len(df) == 0:
        return None
    dates = df[date_col].dt.strftime("%Y-%m-%d").tolist()
    log_returns = df["log_return_1"].to_numpy(dtype=float)
    return dates, log_returns


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
        df["time"] = pd.to_datetime(df["time"])
    df["item_id"] = commodity
    return df


def _build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    anchor = pd.Timestamp("2000-01-01")
    global_times = sorted(df["time"].unique())
    time2idx = {t: i for i, t in enumerate(global_times)}
    df = df.copy()
    df["time_idx_int"] = df["time"].map(time2idx)
    df["time_idx"] = anchor + pd.to_timedelta(df["time_idx_int"].astype(int), unit="D")
    return df


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["rv_5"] = df["log_return_1"].rolling(5).std()
    df["rv_10"] = df["log_return_1"].rolling(10).std()

    for col in ["open", "high", "EMA", "EMA_5", "EMA_10"]:
        if col in df.columns:
            df[col] = df[col] / df["close"]

    SENT_SCORES = ["sentiment_score_mean", "sentiment_score_std", "sentiment_score_max", "sentiment_score_min"]
    TIME_SCORES = ["timeframe_score_mean", "timeframe_score_std", "timeframe_score_max", "timeframe_score_min"]
    SENT_RATIOS = ["sentiment_neg_ratio", "sentiment_neu_ratio", "sentiment_pos_ratio"]
    TIME_RATIOS = ["time_past_ratio", "time_present_ratio", "time_future_ratio"]

    feature_cols = [
        c
        for c in df.columns
        if c not in ["time", "item_id", "close", "time_idx", "time_idx_int", "EMA_20", "EMA_50", "EMA_100", "Volume"]
        and not c.startswith("log_return_")
    ]
    feature_cols = [c for c in feature_cols if c not in SENT_RATIOS + TIME_RATIOS + SENT_SCORES + TIME_SCORES]

    # Avoid NaNs/Infs breaking StudentT loss
    numeric_cols = [c for c in (feature_cols + ["log_return_1"]) if c in df.columns]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df, feature_cols


def _log_return_to_close(log_returns: np.ndarray, base_close: float) -> np.ndarray:
    close = np.zeros_like(log_returns, dtype=float)
    close[0] = base_close * np.exp(log_returns[0])
    for i in range(1, len(log_returns)):
        close[i] = close[i - 1] * np.exp(log_returns[i])
    return close


def _load_close_map_from_file(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None or "close" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "close"])
    return dict(zip(df["date"].tolist(), df["close"].tolist()))


def _calendar_from_frames(
    infer_raw: pd.DataFrame | None,
    train_raw: pd.DataFrame | None,
    future_path: Path | None,
) -> List[str]:
    dates: List[str] = []
    for df in [infer_raw, train_raw]:
        if df is None:
            continue
        if "time" in df.columns:
            dates.extend(pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
    if future_path is not None and future_path.exists():
        fdf = pd.read_csv(future_path)
        if "time" in fdf.columns:
            dates.extend(pd.to_datetime(fdf["time"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
        elif "trade_date" in fdf.columns:
            dates.extend(pd.to_datetime(fdf["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
        elif "date" in fdf.columns:
            dates.extend(pd.to_datetime(fdf["date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
    dates = [d for d in dates if d and d != "NaT"]
    return sorted(set(dates))


def _next_trading_dates(
    as_of: str,
    horizons: List[int],
    trading_calendar: List[str],
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if trading_calendar and as_of in trading_calendar:
        idx = trading_calendar.index(as_of)
        future = trading_calendar[idx + 1 :]
        for h in horizons:
            if h - 1 < len(future):
                mapping[h] = future[h - 1]
    base = pd.to_datetime(as_of)
    for h in horizons:
        if h not in mapping:
            mapping[h] = (base + BDay(h)).strftime("%Y-%m-%d")
    return mapping


def _eval_metrics(
    predictor,
    dataset,
    *,
    horizons: List[int],
    num_samples: int,
    quantiles: List[float],
) -> Dict[str, Dict[str, float]]:
    forecast_it, ts_it = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=num_samples)
    forecasts = list(forecast_it)
    tss = list(ts_it)

    q50 = 0.5 if 0.5 in quantiles else quantiles[len(quantiles) // 2]
    q10 = 0.1 if 0.1 in quantiles else min(quantiles)
    q90 = 0.9 if 0.9 in quantiles else max(quantiles)

    preds_by_h = {h: [] for h in horizons}
    trues_by_h = {h: [] for h in horizons}
    q10_by_h = {h: [] for h in horizons}
    q90_by_h = {h: [] for h in horizons}

    all_preds = []
    all_trues = []
    all_q10 = []
    all_q90 = []

    for fc, ts in zip(forecasts, tss):
        start = fc.start_date
        # slice true values for prediction range
        try:
            true_vals = ts[start : start + fc.prediction_length - 1].to_numpy()
        except Exception:
            true_vals = ts[-fc.prediction_length :].to_numpy()
        if len(true_vals) < fc.prediction_length:
            continue

        pred_vals = fc.quantile(q50)
        q10_vals = fc.quantile(q10)
        q90_vals = fc.quantile(q90)

        for h in horizons:
            idx = h - 1
            if idx >= len(true_vals):
                continue
            preds_by_h[h].append(float(pred_vals[idx]))
            trues_by_h[h].append(float(true_vals[idx]))
            q10_by_h[h].append(float(q10_vals[idx]))
            q90_by_h[h].append(float(q90_vals[idx]))

        all_preds.extend(pred_vals.tolist())
        all_trues.extend(true_vals.tolist())
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

    return {
        "overall": overall,
        "per_horizon": per_h,
    }


def _eval_metrics_rolling(
    df: pd.DataFrame,
    predictor,
    *,
    target_col: str,
    feature_cols: List[str],
    context_length: int,
    prediction_length: int,
    horizons: List[int],
    num_samples: int,
    quantiles: List[float],
) -> Dict[str, Dict[str, float]]:
    if len(df) < context_length + prediction_length:
        return {"skipped": True, "reason": "val_too_short"}

    df = df.sort_values("time_idx").reset_index(drop=True)

    q50 = 0.5 if 0.5 in quantiles else quantiles[len(quantiles) // 2]
    q10 = 0.1 if 0.1 in quantiles else min(quantiles)
    q90 = 0.9 if 0.9 in quantiles else max(quantiles)

    preds_by_h = {h: [] for h in horizons}
    trues_by_h = {h: [] for h in horizons}
    q10_by_h = {h: [] for h in horizons}
    q90_by_h = {h: [] for h in horizons}

    all_preds = []
    all_trues = []
    all_q10 = []
    all_q90 = []

    start = df["time_idx"].iloc[0]

    for i in range(context_length - 1, len(df) - prediction_length):
        target = df[target_col].iloc[: i + 1].to_numpy(dtype=float)
        feats = np.vstack([df[c].iloc[: i + 1 + prediction_length].to_numpy(dtype=float) for c in feature_cols])
        dataset = ListDataset(
            [{"target": target, "start": start, "feat_dynamic_real": feats}],
            freq="D",
        )
        forecast = next(predictor.predict(dataset, num_samples=num_samples))
        true_future = df[target_col].iloc[i + 1 : i + 1 + prediction_length].to_numpy(dtype=float)
        if len(true_future) < prediction_length:
            continue

        pred_vals = forecast.quantile(q50)
        q10_vals = forecast.quantile(q10)
        q90_vals = forecast.quantile(q90)

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


def _load_val_dates(split_file: str, fold_idx: int) -> List[str]:
    with open(split_file, "r") as f:
        data = json.load(f)
    val_dates = data["folds"][fold_idx]["val"]["t_dates"]
    return [str(pd.to_datetime(d).date()) for d in val_dates]


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
        if len(preds_by_h[h]) == 0:
            continue
        per_h[str(h)] = _metrics(
            np.array(preds_by_h[h], dtype=float),
            np.array(trues_by_h[h], dtype=float),
            np.array(q10_by_h[h], dtype=float),
            np.array(q90_by_h[h], dtype=float),
        )

    return {
        "overall": _metrics(all_preds, all_trues, all_q10, all_q90),
        "per_horizon": per_h,
    }


def _build_output_roots(cfg: DeepARConfig, as_of: str) -> tuple[Path, Path, Path]:
    tag = f"_{cfg.output_tag}" if cfg.output_tag else ""
    ckpt_root = Path(cfg.checkpoint_root.format(commodity=cfg.target_commodity, date=as_of, tag=tag))
    pred_root = Path(cfg.prediction_root.format(commodity=cfg.target_commodity, date=as_of, tag=tag))
    combined = Path(cfg.combined_output.format(commodity=cfg.target_commodity, date=as_of, tag=tag))
    return ckpt_root, pred_root, combined


def _infer_as_of(df: pd.DataFrame) -> str:
    return str(pd.to_datetime(df["time"]).max())[:10]


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


def main(cfg: DeepARConfig) -> None:
    set_seed(cfg.seed)
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
                "Please ensure train_price.csv and inference_price.csv do not overlap."
            )
        df = df.sort_values("time")
    else:
        df = train_raw.copy()

    df, feature_cols = _prepare_features(df)
    df = _build_time_index(df)

    # split (train/val) from json
    split_file = cfg.split_file.format(commodity=cfg.target_commodity)
    train_df, val_df = deepar_split(df, split_file, cfg.fold[0])
    val_dates = _load_val_dates(split_file, cfg.fold[0])

    # inference series: use full combined series (train + inference if present)
    full_df = df.copy()

    if infer_raw is not None and not infer_raw.empty:
        as_of = _infer_as_of(infer_raw)
    else:
        as_of = _infer_as_of(train_raw)
    future_payload = None
    if cfg.do_test:
        future_payload = _load_future_log_return(
            cfg.data_dir,
            cfg.target_commodity,
            as_of,
            cfg.prediction_length,
        )
    future_path = _resolve_future_price_path(cfg.data_dir, cfg.target_commodity, cfg.future_price_file)
    trading_calendar = _calendar_from_frames(infer_raw, train_raw, future_path)
    ckpt_root, pred_root, combined_out = _build_output_roots(cfg, as_of)

    combined_rows = []

    for ctx_len in cfg.seq_lengths:
        print(f"\n▶ DeepAR window={ctx_len}")
        w_ckpt = ckpt_root / f"w{ctx_len}"
        w_ckpt.mkdir(parents=True, exist_ok=True)

        # build datasets
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

        # metrics
        # validation metrics (TFT-style: use all val dates with rolling context)
        df_for_val = df[df["time"].isin(train_raw["time"])].reset_index(drop=True)
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
        )

        # inference
        infer_ds = _build_infer_dataset(full_df, "log_return_1", feature_cols, cfg.prediction_length)
        forecast_it = predictor.predict(infer_ds, num_samples=cfg.num_samples)
        forecast = list(forecast_it)[0]

        base_close = float(full_df.sort_values("time").iloc[-1]["close"])

        quantile_close = {}
        for q in cfg.quantiles:
            log_q = forecast.quantile(q)
            quantile_close[f"q{int(q*100):02d}"] = _log_return_to_close(log_q, base_close)

        # test metrics (log_return-based using test_price if available)
        test_metrics = None
        if future_payload is not None:
            future_dates, future_log_returns = future_payload
            q50 = 0.5 if 0.5 in cfg.quantiles else cfg.quantiles[len(cfg.quantiles) // 2]
            q10 = 0.1 if 0.1 in cfg.quantiles else min(cfg.quantiles)
            q90 = 0.9 if 0.9 in cfg.quantiles else max(cfg.quantiles)
            q50_vals = forecast.quantile(q50)
            q10_vals = forecast.quantile(q10)
            q90_vals = forecast.quantile(q90)
            n = min(len(future_log_returns), len(q50_vals))
            if n == 0:
                err = None
            else:
                err = q50_vals[:n] - future_log_returns[:n]
            if err is not None:
                future_dates = future_dates[: len(err)]
                future_log_returns = future_log_returns[: len(err)]
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err ** 2)))
                mape = float(np.mean(np.abs(err / (future_log_returns + 1e-12))) * 100)
                ss_res = float(np.sum((future_log_returns - q50_vals[: len(err)]) ** 2))
                ss_tot = float(np.sum((future_log_returns - np.mean(future_log_returns)) ** 2))
                r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
                da = float(
                    np.mean(
                        (q50_vals[: len(err)] > 0) == (future_log_returns > 0)
                    )
                ) if len(future_log_returns) > 1 else float("nan")
                coverage = float(
                    np.mean((future_log_returns >= q10_vals[: len(err)]) & (future_log_returns <= q90_vals[: len(err)]))
                )
                per_h = {}
                for h in cfg.horizons:
                    idx = h - 1
                    if idx >= len(err):
                        continue
                    per_h[str(h)] = {
                        "MAE": float(abs(err[idx])),
                        "RMSE": float(abs(err[idx])),
                        "MAPE": float(abs(err[idx] / (future_log_returns[idx] + 1e-12)) * 100),
                    }
                test_metrics = {
                    "source": "test_price",
                    "as_of": as_of,
                    "test_range": {
                        "start": future_dates[0],
                        "end": future_dates[-1],
                        "n": len(future_dates),
                    },
                    "overall": {
                        "MAE": mae,
                        "RMSE": rmse,
                        "MAPE": mape,
                        "DA": da,
                        "R2": r2,
                        "coverage_q10_q90": coverage,
                    },
                    "per_horizon": per_h,
                }

        if test_metrics is None:
            if cfg.do_test:
                test_metrics = {
                    "skipped": True,
                    "reason": "future_price_not_available",
                }
            else:
                test_metrics = {
                    "skipped": True,
                    "reason": "do_test_false",
                }

        val_payload = {
            "model": "deepar",
            "commodity": cfg.target_commodity,
            "fold": cfg.fold[0],
            "seq_length": ctx_len,
            "horizons": cfg.horizons,
        }
        if isinstance(val_metrics, dict) and val_metrics.get("skipped"):
            val_payload.update(val_metrics)
        else:
            val_payload.update(
                {
                    "overall": val_metrics.get("overall"),
                    "per_horizon": val_metrics.get("per_horizon"),
                }
            )

        test_payload = {
            "model": "deepar",
            "commodity": cfg.target_commodity,
            "fold": cfg.fold[0],
            "seq_length": ctx_len,
            "horizons": cfg.horizons,
        }
        if isinstance(test_metrics, dict) and test_metrics.get("skipped"):
            test_payload.update(test_metrics)
        else:
            test_payload.update(test_metrics)
        (w_ckpt / "val_metrics.json").write_text(json.dumps(val_payload, indent=2))
        (w_ckpt / "test_metrics.json").write_text(json.dumps(test_payload, indent=2))

        # save plot
        w_pred_root = pred_root / f"w{ctx_len}"
        w_pred_root.mkdir(parents=True, exist_ok=True)
        x = np.arange(1, cfg.prediction_length + 1)
        q10 = quantile_close.get("q10")
        q50 = quantile_close.get("q50")
        q90 = quantile_close.get("q90")
        if q10 is not None and q50 is not None and q90 is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(x, q50, color="tab:blue", label="q50")
            plt.fill_between(x, q10, q90, color="tab:blue", alpha=0.2, label="q10-q90")
            plt.title(f"{cfg.target_commodity} Close Forecast (w{ctx_len})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(w_pred_root / f"{as_of}_forecast_q10_q90.png", dpi=150)
            plt.close()

        # combined csv rows (TFT-style)
        date_map = _next_trading_dates(as_of, cfg.horizons, trading_calendar)
        q50 = quantile_close.get("q50")
        q10 = quantile_close.get("q10")
        q90 = quantile_close.get("q90")
        if q50 is None:
            # fallback to median of available quantiles
            qkeys = sorted(quantile_close.keys())
            q50 = quantile_close[qkeys[len(qkeys) // 2]]
        if q10 is None:
            q10 = q50
        if q90 is None:
            q90 = q50
        for h in cfg.horizons:
            if h - 1 >= len(q50):
                continue
            predict_date = date_map.get(h)
            if predict_date is None:
                continue
            combined_rows.append(
                {
                    "date": as_of,
                    "window": ctx_len,
                    "predict_date": predict_date,
                    "predicted_close": float(q50[h - 1]),
                    "predicted_q10": float(q10[h - 1]),
                    "predicted_q90": float(q90[h - 1]),
                }
            )

    if combined_rows:
        combined_out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["date", "window", "predict_date", "predicted_close", "predicted_q10", "predicted_q90"]
        df_out = pd.DataFrame(combined_rows)
        df_out = df_out[cols]
        df_out.to_csv(combined_out, index=False)
        print(f"\n✓ Combined predictions saved: {combined_out}")


if __name__ == "__main__":
    main(tyro.cli(DeepARConfig))
