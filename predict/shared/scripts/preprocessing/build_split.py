"""
Build a simple time-based split JSON for training/eval.

Usage:
python scripts/build_split.py \
  --data_source bigquery \
  --bq_project_id esoteric-buffer-485608-g5 \
  --bq_dataset_id final_proj \
  --bq_train_table train_price \
  --target_commodity corn \
  --val_months 3 \
  --output_dir shared/src/datasets/bq_splits

Local example:
python scripts/build_split.py \
  --data_source local \
  --data_dir shared/src/datasets/local_bq_like/corn \
  --target_commodity corn \
  --val_months 3 \
  --output_dir shared/src/datasets/local_splits/eval
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import sys
from typing import Optional, Dict

import numpy as np
import pandas as pd
import tyro

SHARED_ROOT = Path(__file__).resolve().parents[2]
PREDICT_ROOT = SHARED_ROOT.parent
if str(PREDICT_ROOT) not in sys.path:
    sys.path.insert(0, str(PREDICT_ROOT))
REPO_ROOT = PREDICT_ROOT.parent

from shared.data.bigquery_loader import load_price_table


@dataclass
class SplitConfig:
    data_source: str = "bigquery"  # bigquery or local
    data_dir: str = "shared/src/datasets"
    target_commodity: str = "corn"
    val_months: int = 2
    output_file: str = ""
    output_dir: str = ""
    test_days: int = 0
    test_end_date: str = ""
    inference_days: int = 0
    train_years: int = 0
    extra_days: int = 0
    target_horizon: int = 20
    write_price_splits: bool = False
    drop_log_return_in_inference: bool = True

    # BigQuery
    bq_project_id: str = ""
    bq_dataset_id: str = ""
    bq_train_table: str = "train_price"

    # Local fallback
    local_price_file: str = ""
    prefer_feature_engineering: bool = False


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


def _resolve_price_bq_settings(cfg: SplitConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("PRICE_BIGQUERY_PROJECT") or cfg.bq_project_id,
        "dataset_id": _get("PRICE_BIGQUERY_DATASET") or cfg.bq_dataset_id,
        "credentials_path": _get("PRICE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def _load_price_df(cfg: SplitConfig) -> pd.DataFrame:
    if cfg.data_source == "bigquery":
        bq = _resolve_price_bq_settings(cfg)
        if bq.get("credentials_path") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = bq["credentials_path"]
        return load_price_table(
            project_id=bq.get("project_id") or cfg.bq_project_id,
            dataset_id=bq.get("dataset_id") or cfg.bq_dataset_id,
            table=cfg.bq_train_table,
            commodity=cfg.target_commodity,
        )

    local_file = cfg.local_price_file
    if not local_file:
        # For eval/test splits, use full feature_engineering to keep tail dates.
        if cfg.test_days > 0 or cfg.prefer_feature_engineering:
            local_file = f"preprocessing/{cfg.target_commodity}_feature_engineering.csv"
        else:
            # Prefer train_price.csv if present (deploy-style)
            train_price = Path(cfg.data_dir) / "train_price.csv"
            if train_price.exists():
                local_file = "train_price.csv"
            else:
                local_file = f"preprocessing/{cfg.target_commodity}_feature_engineering.csv"
    price_path = Path(cfg.data_dir) / local_file
    if not price_path.exists():
        # Fallback to global preprocessing dir: shared/src/datasets/preprocessing/{commodity}_feature_engineering.csv
        fallback = Path(cfg.data_dir).parents[1] / "preprocessing" / f"{cfg.target_commodity}_feature_engineering.csv"
        if fallback.exists():
            price_path = fallback
        else:
            raise FileNotFoundError(f"Local price file not found: {price_path}")
    df = pd.read_csv(price_path)
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in local price file.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def build_split(cfg: SplitConfig) -> dict:
    df = _load_price_df(cfg)
    df = df.dropna(subset=["time"]).sort_values("time")
    # Drop rows without target horizon (e.g., tail rows missing log_return_20)
    target_col = f"log_return_{cfg.target_horizon}"
    if target_col in df.columns and cfg.test_days <= 0:
        df = df[df[target_col].notna()].copy()
    dates = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d").unique().tolist()

    if not dates:
        raise ValueError("No dates found to build split.")

    all_dates = dates
    test_dates: list[str] = []
    inference_dates: list[str] = []
    train_pool = all_dates

    if cfg.test_days and cfg.test_days > 0:
        if len(all_dates) <= cfg.test_days:
            raise ValueError("Not enough dates to hold out test_days.")
        if cfg.test_end_date:
            if cfg.test_end_date not in all_dates:
                raise ValueError(f"test_end_date {cfg.test_end_date} not found in dates.")
            end_idx = all_dates.index(cfg.test_end_date)
        else:
            end_idx = len(all_dates) - 1
        start_idx = end_idx - cfg.test_days + 1
        if start_idx < 0:
            raise ValueError("Not enough dates to build test window.")
        test_dates = all_dates[start_idx : end_idx + 1]
        train_pool = all_dates[:start_idx]
        if cfg.inference_days and cfg.inference_days > 0:
            if start_idx - cfg.inference_days < 0:
                raise ValueError("Not enough dates to build inference window.")
            inf_start = start_idx - cfg.inference_days
            inference_dates = all_dates[inf_start:start_idx]
            train_pool = all_dates[:inf_start]

    if not train_pool:
        raise ValueError("Split failed: empty train pool.")

    train_end = pd.to_datetime(train_pool[-1])
    if cfg.train_years or cfg.extra_days:
        train_start = train_end - pd.DateOffset(years=cfg.train_years) + pd.Timedelta(days=1)
        if cfg.extra_days and cfg.extra_days > 0:
            train_start = train_start - pd.Timedelta(days=cfg.extra_days)
        train_pool = [d for d in train_pool if pd.to_datetime(d) >= train_start]

    if not train_pool:
        raise ValueError("Split failed: empty train pool after windowing.")

    val_start = (train_end - pd.DateOffset(months=cfg.val_months)) + pd.Timedelta(days=1)
    val_dates = [d for d in train_pool if pd.to_datetime(d) >= val_start]
    train_dates = [d for d in train_pool if pd.to_datetime(d) < val_start]

    if not train_dates or not val_dates:
        raise ValueError("Split failed: empty train or validation dates.")

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    train_indices = [date_to_idx[d] for d in train_dates]
    val_indices = [date_to_idx[d] for d in val_dates]
    test_indices = [date_to_idx[d] for d in test_dates] if test_dates else []

    meta = {
        "split_type": "recent_months",
        "val_months": cfg.val_months,
        "start_date": all_dates[0],
        "end_date": all_dates[-1],
        "n_days": len(all_dates),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if test_dates:
        meta["test_days"] = cfg.test_days
        if cfg.test_end_date:
            meta["test_end_date"] = cfg.test_end_date
        meta["fixed_test"] = {"t_dates": test_dates, "t_indices": test_indices}
    if inference_dates:
        meta["inference_days"] = cfg.inference_days
        meta["inference_window"] = {
            "start": inference_dates[0],
            "end": inference_dates[-1],
        }

    split = {
        "meta": {
            **meta
        },
        "folds": [
            {
                "fold": 0,
                "train": {"t_dates": train_dates, "t_indices": train_indices},
                "val": {"t_dates": val_dates, "t_indices": val_indices},
            }
        ],
    }

    if cfg.write_price_splits and cfg.data_source == "local":
        _write_price_splits(
            cfg=cfg,
            df=df,
            train_pool_dates=train_dates + val_dates,
            inference_dates=inference_dates,
            test_dates=test_dates,
        )

    return split


def _write_price_splits(
    cfg: SplitConfig,
    df: pd.DataFrame,
    train_pool_dates: list[str],
    inference_dates: list[str],
    test_dates: list[str],
) -> None:
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column for price split write.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    date_str = df["time"].dt.strftime("%Y-%m-%d")

    # Ensure log_return_1..target_horizon exist (computed from close)
    _ensure_log_returns(df, cfg.target_horizon)

    train_df = df[date_str.isin(train_pool_dates)].copy()
    train_df = train_df.sort_values("time")

    out_dir = Path(cfg.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_price.csv"
    train_df.to_csv(train_path, index=False)

    if test_dates:
        test_df = df[date_str.isin(test_dates)].copy()
        test_df = test_df.sort_values("time")
        test_path = out_dir / "test_price.csv"
        test_df.to_csv(test_path, index=False)

    if inference_dates:
        inf_df = df[date_str.isin(inference_dates)].copy()
        inf_df = inf_df.sort_values("time")
        if cfg.drop_log_return_in_inference:
            log_cols = [c for c in inf_df.columns if c.startswith("log_return_")]
            if log_cols:
                inf_df = inf_df.drop(columns=log_cols)
        inf_path = out_dir / "inference_price.csv"
        inf_df.to_csv(inf_path, index=False)


def _ensure_log_returns(df: pd.DataFrame, max_horizon: int) -> None:
    if "close" not in df.columns:
        raise ValueError("Expected 'close' column to compute log_return horizons.")
    close = df["close"].astype(float)
    for h in range(1, max_horizon + 1):
        df[f"log_return_{h}"] = np.log(close.shift(-h) / close)


def _normalize_shared_path(path_str: str) -> str:
    if not path_str:
        return path_str
    if "shared/datasets" in path_str and "shared/src/datasets" not in path_str:
        return path_str.replace("shared/datasets", "shared/src/datasets")
    return path_str


def main(cfg: SplitConfig) -> None:
    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = PREDICT_ROOT / path
        return str(path)

    cfg.data_dir = _normalize_shared_path(_resolve_path(cfg.data_dir))
    if cfg.output_dir:
        cfg.output_dir = _normalize_shared_path(_resolve_path(cfg.output_dir))
    if cfg.output_file:
        cfg.output_file = _normalize_shared_path(_resolve_path(cfg.output_file))
    if cfg.local_price_file:
        cfg.local_price_file = _normalize_shared_path(_resolve_path(cfg.local_price_file))

    split = build_split(cfg)
    if cfg.output_file:
        output_path = Path(cfg.output_file)
    elif cfg.output_dir:
        output_path = Path(cfg.output_dir) / f"{cfg.target_commodity}_split.json"
    else:
        # Default to {commodity}_split.json under data_dir for local eval usage.
        output_path = Path(cfg.data_dir) / f"{cfg.target_commodity}_split.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2))
    print(f"Split saved to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(SplitConfig))
