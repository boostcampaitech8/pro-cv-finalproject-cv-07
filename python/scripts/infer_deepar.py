"""
DeepAR inference only (load saved predictor and export quantile CSV/PNG).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import tyro

from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import ListDataset

from src.data.dataset import build_multi_item_dataset
from src.data.bigquery_loader import load_price_table


@dataclass
class DeepARInferConfig:
    data_dir: str = "src/datasets/local_bq_like/corn"
    target_commodity: str = "corn"
    data_source: str = "local"  # local or bigquery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"
    future_price_file: str = "src/datasets/{commodity}_future_price.csv"
    seq_length: int = 20
    prediction_length: int = 20
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    num_samples: int = 200
    checkpoint_dir: str = "src/outputs/checkpoints/{commodity}_{date}_deepar/w{seq}/best_model"
    output_root: str = "src/outputs/predictions/{commodity}_{date}_deepar"


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
    horizons: List[int],
    exchange: str = "NYSE",
) -> dict[int, str]:
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
    mapping: dict[int, str] = {}
    for h in horizons:
        idx = int(h) - 1
        if 0 <= idx < len(day_list):
            mapping[int(h)] = day_list[idx]
    return mapping


def _next_trading_dates(
    as_of: str,
    horizons: List[int],
    trading_calendar: List[str],
    exchange: str = "NYSE",
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if trading_calendar and as_of in trading_calendar:
        idx = trading_calendar.index(as_of)
        future = trading_calendar[idx + 1 :]
        for h in horizons:
            if h - 1 < len(future):
                mapping[h] = future[h - 1]
    missing = [int(h) for h in horizons if int(h) not in mapping]
    if missing:
        mapping.update(_exchange_future_dates(as_of, missing, exchange=exchange))
    base = pd.to_datetime(as_of)
    for h in horizons:
        if h not in mapping:
            mapping[h] = (base + BDay(h)).strftime("%Y-%m-%d")
    return mapping


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


def main(cfg: DeepARInferConfig) -> None:
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
        df = pd.concat([train_raw, infer_raw], axis=0).drop_duplicates(subset=["time"]).sort_values("time")
    else:
        df = train_raw.copy()

    df = _ensure_log_return1(df)
    df, feature_cols = _prepare_features(df)
    df = _build_time_index(df)
    if infer_raw is not None and not infer_raw.empty:
        as_of = str(pd.to_datetime(infer_raw["time"]).max())[:10]
    else:
        as_of = str(pd.to_datetime(df["time"]).max())[:10]

    ckpt_dir = Path(
        cfg.checkpoint_dir.format(
            commodity=cfg.target_commodity,
            date=as_of,
            seq=cfg.seq_length,
        )
    )
    predictor = PyTorchPredictor.deserialize(ckpt_dir)
    infer_source = df
    if infer_raw is not None and not infer_raw.empty:
        infer_start = pd.to_datetime(infer_raw["time"]).min()
        infer_block = df[df["time"] >= infer_start].copy()
        if len(infer_block) < cfg.seq_length:
            pad_len = cfg.seq_length - len(infer_block)
            train_tail = df[df["time"] < infer_start].tail(pad_len)
            infer_source = pd.concat([train_tail, infer_block], ignore_index=True)
        else:
            infer_source = infer_block

    infer_ds = _build_infer_dataset(infer_source, "log_return_1", feature_cols, cfg.prediction_length)
    forecast_it = predictor.predict(infer_ds, num_samples=cfg.num_samples)
    forecast = list(forecast_it)[0]

    base_close = float(infer_source.sort_values("time").iloc[-1]["close"])
    quantile_close = {}
    for q in cfg.quantiles:
        log_q = forecast.quantile(q)
        quantile_close[f"q{int(q*100):02d}"] = _log_return_to_close(log_q, base_close)

    out_root = Path(cfg.output_root.format(commodity=cfg.target_commodity, date=as_of))
    w_root = out_root / f"w{cfg.seq_length}"
    w_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    combined_csv = out_root / "deepar_predictions.csv"

    future_path = _resolve_future_price_path(cfg.data_dir, cfg.target_commodity, cfg.future_price_file)
    trading_calendar = _calendar_from_frames(infer_raw, train_raw, future_path)
    date_map = _next_trading_dates(
        as_of,
        list(range(1, cfg.prediction_length + 1)),
        trading_calendar,
        exchange=_resolve_exchange_for_commodity(cfg.target_commodity),
    )

    q50 = quantile_close.get("q50")
    q10 = quantile_close.get("q10")
    q90 = quantile_close.get("q90")
    if q50 is None:
        qkeys = sorted(quantile_close.keys())
        q50 = quantile_close[qkeys[len(qkeys) // 2]]
    if q10 is None:
        q10 = q50
    if q90 is None:
        q90 = q50

    combined_rows = []
    for h in range(1, cfg.prediction_length + 1):
        predict_date = date_map.get(h)
        if predict_date is None:
            continue
        combined_rows.append(
            {
                "date": as_of,
                "window": cfg.seq_length,
                "predict_date": predict_date,
                "predicted_close": float(q50[h - 1]),
                "predicted_q10": float(q10[h - 1]),
                "predicted_q90": float(q90[h - 1]),
            }
        )
    df_combined = pd.DataFrame(combined_rows)
    if combined_csv.exists():
        df_prev = pd.read_csv(combined_csv)
        df_prev = df_prev[~((df_prev["date"] == as_of) & (df_prev["window"] == cfg.seq_length))]
        df_combined = pd.concat([df_prev, df_combined], ignore_index=True)
    df_combined = df_combined[
        ["date", "window", "predict_date", "predicted_close", "predicted_q10", "predicted_q90"]
    ]
    df_combined.to_csv(combined_csv, index=False)

    # plot
    import matplotlib.pyplot as plt
    x = np.arange(1, cfg.prediction_length + 1)
    q10 = quantile_close.get("q10")
    q50 = quantile_close.get("q50")
    q90 = quantile_close.get("q90")
    if q10 is not None and q50 is not None and q90 is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(x, q50, color="tab:blue", label="q50")
        plt.fill_between(x, q10, q90, color="tab:blue", alpha=0.2, label="q10-q90")
        plt.title(f"{cfg.target_commodity} Close Forecast (w{cfg.seq_length})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(w_root / f"{as_of}_forecast_q10_q90.png", dpi=150)
        plt.close()

    print(f"✓ DeepAR inference saved: {combined_csv}")


if __name__ == "__main__":
    main(tyro.cli(DeepARInferConfig))
