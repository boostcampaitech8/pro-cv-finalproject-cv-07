"""
Build rolling split JSONs from BigQuery price tables (train/inference) per symbol.

Outputs:
  - {commodity}_split.json

Example:
python scripts/build_bq_split.py \
  --project_id esoteric-buffer-485608-g5 \
  --dataset_id final_proj \
  --commodities corn wheat \
  --val_months 3 \
  --test_days 20 \
  --output_dir src/datasets/bq_splits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import json

import pandas as pd
import tyro

sys.path.append(str(Path(__file__).parent.parent))

from src.data.bigquery_loader import COMMODITY_TO_SYMBOL


SYMBOL_TO_COMMODITY = {v: k for k, v in COMMODITY_TO_SYMBOL.items()}


@dataclass
class BQSplitConfig:
    project_id: str
    dataset_id: str
    train_table: str = "train_price"
    inference_table: str = "inference_price"

    output_dir: str = "src/datasets/bq_splits"

    # Provide either commodities (preferred) or symbols
    commodities: List[str] = field(default_factory=lambda: ["corn"])
    symbols: List[str] = field(default_factory=list)

    val_months: int = 3
    test_days: int = 20
    write_default_split: bool = False

    # Optional cap for end date (YYYY-MM-DD)
    end_date: Optional[str] = None
    # Optional sanity check that inference table has rows for the symbol
    check_inference: bool = True


def _build_split(dates: list[str], val_months: int, test_days: int) -> dict:
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

    train_dates: list[str] = []
    val_dates: list[str] = []
    test_dates: list[str] = []

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


def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("=", "").replace("-", "_").lower()


def _resolve_targets(cfg: BQSplitConfig) -> List[Tuple[str, str]]:
    if cfg.symbols:
        targets: List[Tuple[str, str]] = []
        for sym in cfg.symbols:
            commodity = SYMBOL_TO_COMMODITY.get(sym, _sanitize_symbol(sym))
            targets.append((commodity, sym))
        return targets

    targets = []
    for commodity in cfg.commodities:
        symbol = COMMODITY_TO_SYMBOL.get(commodity.lower())
        if not symbol:
            raise ValueError(f"Unknown commodity '{commodity}'. Add it to COMMODITY_TO_SYMBOL.")
        targets.append((commodity.lower(), symbol))
    return targets


def _fetch_dates(
    client,
    project_id: str,
    dataset_id: str,
    table: str,
    symbol: str,
    end_date: Optional[str],
) -> list[str]:
    where = [f"symbol = '{symbol}'"]
    if end_date:
        where.append(f"trade_date <= '{end_date}'")
    where_clause = " WHERE " + " AND ".join(where)
    query = (
        f"SELECT trade_date FROM `{project_id}.{dataset_id}.{table}`"
        f"{where_clause} ORDER BY trade_date"
    )
    df = client.query(query).to_dataframe()
    if "trade_date" not in df.columns:
        raise ValueError(f"{table} must contain trade_date column.")
    dates = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
    return dates.dt.strftime("%Y-%m-%d").unique().tolist()


def _check_inference_rows(client, project_id: str, dataset_id: str, table: str, symbol: str) -> int:
    query = (
        f"SELECT COUNT(1) as n FROM `{project_id}.{dataset_id}.{table}` "
        f"WHERE symbol = '{symbol}'"
    )
    df = client.query(query).to_dataframe()
    return int(df.iloc[0]["n"]) if not df.empty else 0


def main(cfg: BQSplitConfig) -> None:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError("google-cloud-bigquery is required to run build_bq_split.py") from exc

    client = bigquery.Client(project=cfg.project_id)
    targets = _resolve_targets(cfg)

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for commodity, symbol in targets:
        dates = _fetch_dates(
            client=client,
            project_id=cfg.project_id,
            dataset_id=cfg.dataset_id,
            table=cfg.train_table,
            symbol=symbol,
            end_date=cfg.end_date,
        )
        if not dates:
            print(f"[WARN] No dates found for {symbol} in {cfg.train_table}.")
            continue

        split = _build_split(dates, cfg.val_months, cfg.test_days)
        split["meta"]["commodity"] = commodity
        split["meta"]["symbol"] = symbol
        if cfg.end_date:
            split["meta"]["end_date_cap"] = cfg.end_date

        out_path = output_root / f"{commodity}_split.json"
        out_path.write_text(json.dumps(split, indent=2))

        if cfg.write_default_split:
            default_path = output_root / f"{commodity}_split.json"
            default_path.write_text(json.dumps(split, indent=2))

            if len(targets) == 1:
                # Optional convenience alias
                alias_path = output_root / "rolling_fold.json"
                alias_path.write_text(json.dumps(split, indent=2))

        if cfg.check_inference:
            n_infer = _check_inference_rows(
                client=client,
                project_id=cfg.project_id,
                dataset_id=cfg.dataset_id,
                table=cfg.inference_table,
                symbol=symbol,
            )
            if n_infer == 0:
                print(f"[WARN] No inference rows for {symbol} in {cfg.inference_table}.")

        print(f"✓ {commodity} ({symbol}) splits written to {output_root}")


if __name__ == "__main__":
    main(tyro.cli(BQSplitConfig))
