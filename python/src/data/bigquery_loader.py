from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

COMMODITY_TO_SYMBOL = {
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soybean": "ZS=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
}


def _build_where(symbols: Optional[Iterable[str]] = None, start_date: Optional[str] = None) -> str:
    clauses = []
    if symbols:
        symbol_list = ", ".join([f"'{s}'" for s in symbols])
        clauses.append(f"symbol IN ({symbol_list})")
    if start_date:
        clauses.append(f"trade_date >= '{start_date}'")
    if not clauses:
        return ""
    return " WHERE " + " AND ".join(clauses)


def load_price_table(
    *,
    project_id: str,
    dataset_id: str,
    table: str,
    commodity: Optional[str] = None,
    start_date: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load price table from BigQuery and normalize to expected schema.
    """
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError("google-cloud-bigquery is required for BigQuery access.") from exc

    client = bigquery.Client(project=project_id)

    symbols = None
    if commodity:
        symbol = COMMODITY_TO_SYMBOL.get(commodity.lower())
        if symbol:
            symbols = [symbol]

    select_cols = "*"
    if columns:
        select_cols = ", ".join(columns)

    where_clause = _build_where(symbols=symbols, start_date=start_date)
    query = f"SELECT {select_cols} FROM `{project_id}.{dataset_id}.{table}`{where_clause} ORDER BY trade_date"

    df = client.query(query).to_dataframe()

    if "trade_date" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"trade_date": "time"})

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df
