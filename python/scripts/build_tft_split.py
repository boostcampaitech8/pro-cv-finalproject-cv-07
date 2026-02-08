"""
Build a simple time-based split JSON for TFT training.

Usage:
python scripts/build_tft_split.py \
  --data_source bigquery \
  --bq_project_id esoteric-buffer-485608-g5 \
  --bq_dataset_id final_proj \
  --bq_train_table train_price \
  --target_commodity corn \
  --val_months 2 \
  --output_file src/datasets/rolling_fold_2m.json
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import tyro

from src.data.bigquery_loader import load_price_table


@dataclass
class SplitConfig:
    data_source: str = "bigquery"  # bigquery or local
    data_dir: str = "src/datasets"
    target_commodity: str = "corn"
    val_months: int = 2
    output_file: str = ""

    # BigQuery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"

    # Local fallback
    local_price_file: str = ""


def _load_price_df(cfg: SplitConfig) -> pd.DataFrame:
    if cfg.data_source == "bigquery":
        return load_price_table(
            project_id=cfg.bq_project_id,
            dataset_id=cfg.bq_dataset_id,
            table=cfg.bq_train_table,
            commodity=cfg.target_commodity,
        )

    local_file = cfg.local_price_file or f"preprocessing/{cfg.target_commodity}_feature_engineering.csv"
    price_path = Path(cfg.data_dir) / local_file
    df = pd.read_csv(price_path)
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in local price file.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def build_split(cfg: SplitConfig) -> dict:
    df = _load_price_df(cfg)
    df = df.dropna(subset=["time"]).sort_values("time")
    dates = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d").unique().tolist()

    if not dates:
        raise ValueError("No dates found to build split.")

    end_date = pd.to_datetime(dates[-1])
    val_start = (end_date - pd.DateOffset(months=cfg.val_months)) + pd.Timedelta(days=1)
    val_dates = [d for d in dates if pd.to_datetime(d) >= val_start]
    train_dates = [d for d in dates if pd.to_datetime(d) < val_start]

    if not train_dates or not val_dates:
        raise ValueError("Split failed: empty train or validation dates.")

    date_to_idx = {d: i for i, d in enumerate(dates)}
    train_indices = [date_to_idx[d] for d in train_dates]
    val_indices = [date_to_idx[d] for d in val_dates]

    return {
        "meta": {
            "split_type": "recent_months",
            "val_months": cfg.val_months,
            "start_date": dates[0],
            "end_date": dates[-1],
            "n_days": len(dates),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fixed_test": {
                "t_dates": []
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


def main(cfg: SplitConfig) -> None:
    split = build_split(cfg)
    if cfg.output_file:
        output_path = Path(cfg.output_file)
    else:
        output_path = (
            Path(cfg.data_dir)
            / f"rolling_fold_{cfg.val_months}m_{cfg.target_commodity}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2))
    print(f"Split saved to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(SplitConfig))
