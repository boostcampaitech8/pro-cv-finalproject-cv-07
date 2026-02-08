"""
Build a local "BigQuery-like" snapshot from preprocessing CSVs for quick TFT/CNN/DeepAR tests.

Outputs:
  - preprocessing/{commodity}_feature_engineering.csv (trimmed, latest window)
  - news_features.csv (copied)
  - train_price.csv (latest 3y + 60d window, excluding inference tail)
  - inference_price.csv (latest N days for inference)
  - rolling_fold_{2m,3m}_{commodity}.json (train/val + fixed test split on train_price dates)

Example:
python scripts/build_local_bq_snapshot.py \
  --target_commodity corn \
  --train_years 3 \
  --extra_days 60 \
  --inference_days 19 \
  --output_dir src/datasets/local_bq_like/corn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
import tyro
from typing import List


@dataclass
class SnapshotConfig:
    data_dir: str = "src/datasets"
    target_commodity: str = "corn"

    price_file: str = "preprocessing/{commodity}_feature_engineering.csv"
    news_file: str = "news_features.csv"

    output_dir: str = "src/datasets/local_bq_like/{commodity}"

    train_years: int = 3
    extra_days: int = 60
    inference_days: int = 19

    val_months_list: List[int] = field(default_factory=lambda: [2, 3])
    test_days: int = 20
    write_default_split: bool = True

    max_horizon: int = 20
    drop_missing_targets: bool = True


def _resolve_path(base: str, template: str, commodity: str) -> Path:
    return Path(base) / template.format(commodity=commodity)


def _normalize_time(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def _build_split(
    dates: list[str],
    val_months: int,
    test_days: int,
) -> dict:
    if not dates:
        raise ValueError("No dates available for split.")

    end_date = pd.to_datetime(dates[-1])
    test_start = None
    if test_days and test_days > 0:
        test_start = end_date - pd.Timedelta(days=test_days - 1)
    if val_months and val_months > 0:
        if test_start is None:
            val_start = (end_date - pd.DateOffset(months=val_months)) + pd.Timedelta(days=1)
        else:
            val_start = (test_start - pd.DateOffset(months=val_months)) + pd.Timedelta(days=1)
    else:
        val_start = None

    train_dates = []
    val_dates = []
    test_dates = []

    for d in dates:
        dt = pd.to_datetime(d)
        if test_start is not None and dt >= test_start:
            test_dates.append(d)
        elif val_start is not None and dt >= val_start:
            val_dates.append(d)
        else:
            train_dates.append(d)

    if not train_dates:
        raise ValueError("Split failed: empty train set.")
    if val_months and not val_dates:
        raise ValueError("Split failed: empty validation set.")

    date_to_idx = {d: i for i, d in enumerate(dates)}
    train_indices = [date_to_idx[d] for d in train_dates]
    val_indices = [date_to_idx[d] for d in val_dates]
    test_indices = [date_to_idx[d] for d in test_dates]

    return {
        "meta": {
            "split_type": "recent_months",
            "val_months": val_months,
            "test_days": test_days,
            "start_date": dates[0],
            "end_date": dates[-1],
            "n_days": len(dates),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fixed_test": {
                "t_dates": test_dates,
                "t_indices": test_indices,
            },
        },
        "folds": [
            {
                "fold": 0,
                "train": {"t_dates": train_dates, "t_indices": train_indices},
                "val": {"t_dates": val_dates, "t_indices": val_indices},
            }
        ],
    }


def main(cfg: SnapshotConfig) -> None:
    price_path = _resolve_path(cfg.data_dir, cfg.price_file, cfg.target_commodity)
    news_path = _resolve_path(cfg.data_dir, cfg.news_file, cfg.target_commodity)

    price_df = pd.read_csv(price_path)
    if "time" not in price_df.columns:
        raise ValueError("price_file must contain 'time' column.")
    if "close" not in price_df.columns:
        raise ValueError("price_file must contain 'close' column for log_return generation.")
    price_df["time"] = _normalize_time(price_df, "time")
    price_df = price_df.dropna(subset=["time"]).sort_values("time")

    # Ensure log_return_1..max_horizon exist for seq-to-seq
    for h in range(1, cfg.max_horizon + 1):
        col = f"log_return_{h}"
        if col not in price_df.columns or price_df[col].isna().all():
            price_df[col] = np.log(price_df["close"].shift(-h) / price_df["close"])
        else:
            # fill missing with computed values if any
            computed = np.log(price_df["close"].shift(-h) / price_df["close"])
            price_df[col] = price_df[col].fillna(computed)

    news_df = pd.read_csv(news_path)
    if "date" not in news_df.columns:
        raise ValueError("news_file must contain 'date' column.")
    news_df["date"] = _normalize_time(news_df, "date")
    news_df = news_df.dropna(subset=["date"]).sort_values("date")

    target_col = f"log_return_{cfg.max_horizon}"
    if target_col not in price_df.columns:
        raise ValueError(f"{target_col} missing in price data.")

    if cfg.inference_days <= 0:
        raise ValueError("inference_days must be > 0.")
    if len(price_df) <= cfg.inference_days:
        raise ValueError("Not enough rows to build inference tail.")

    # Use the last date where target horizon exists (avoid leaking future data)
    valid_mask = price_df[target_col].notna()
    if not valid_mask.any():
        raise ValueError(f"No valid rows found for target {target_col}.")
    last_valid_time = price_df.loc[valid_mask, "time"].max()
    available_df = price_df[price_df["time"] <= last_valid_time].copy()
    if len(available_df) <= cfg.inference_days:
        raise ValueError("Not enough rows (after target cutoff) to build inference tail.")

    inference_df = available_df.tail(cfg.inference_days).copy()
    train_end = available_df.iloc[-cfg.inference_days - 1]["time"]

    train_start = train_end - pd.DateOffset(years=cfg.train_years) + pd.Timedelta(days=1)
    if cfg.extra_days and cfg.extra_days > 0:
        train_start = train_start - pd.Timedelta(days=cfg.extra_days)

    price_snapshot = available_df[available_df["time"] >= train_start].copy()
    train_df = price_snapshot[price_snapshot["time"] <= train_end].copy()

    if cfg.drop_missing_targets:
        train_df = train_df[train_df[target_col].notna()].copy()

    dates = pd.to_datetime(train_df["time"]).dt.strftime("%Y-%m-%d").unique().tolist()
    splits = {m: _build_split(dates, m, cfg.test_days) for m in cfg.val_months_list}

    output_root = Path(cfg.output_dir.format(commodity=cfg.target_commodity))
    prep_dir = output_root / "preprocessing"
    prep_dir.mkdir(parents=True, exist_ok=True)

    price_out = prep_dir / f"{cfg.target_commodity}_feature_engineering.csv"
    news_out = output_root / "news_features.csv"
    train_out = output_root / "train_price.csv"
    infer_out = output_root / "inference_price.csv"

    price_snapshot.to_csv(price_out, index=False)
    news_df.to_csv(news_out, index=False)
    train_df.to_csv(train_out, index=False)
    inference_df.to_csv(infer_out, index=False)
    for months, split in splits.items():
        split_out = output_root / f"rolling_fold_{months}m_{cfg.target_commodity}.json"
        split_out.write_text(json.dumps(split, indent=2))
    if cfg.write_default_split and cfg.val_months_list:
        default_months = cfg.val_months_list[0]
        default_out = output_root / "rolling_fold.json"
        default_out.write_text(json.dumps(splits[default_months], indent=2))

    print("Local snapshot written:")
    print(f"  price: {price_out}")
    print(f"  news: {news_out}")
    print(f"  train_price: {train_out} ({len(train_df)} rows)")
    print(f"  inference_price: {infer_out} ({len(inference_df)} rows)")
    for months in splits:
        print(f"  split: {output_root / f'rolling_fold_{months}m_{cfg.target_commodity}.json'}")
    if cfg.write_default_split and cfg.val_months_list:
        print(f"  split: {output_root / 'rolling_fold.json'}")


if __name__ == "__main__":
    main(tyro.cli(SnapshotConfig))
