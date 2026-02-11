from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

COMMODITY_TO_SYMBOL = {
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soybean": "ZS=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
}

COMMODITY_TO_KEYWORD = {
    "corn": "agriculture",
    "wheat": "agriculture",
    "soybean": "agriculture",
    "gold": "metal",
    "silver": "metal",
    "copper": "metal",
}


def _flatten_to_floats(obj) -> list[float]:
    if obj is None:
        return []
    if isinstance(obj, float) and pd.isna(obj):
        return []
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            try:
                return [float(obj)]
            except Exception:
                return []
    if isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        out: list[float] = []
        for item in obj:
            out.extend(_flatten_to_floats(item))
        return out
    try:
        return [float(obj)]
    except Exception:
        return []


def _ensure_embedding(vec, target_dim: int = 512) -> list[float]:
    vals = _flatten_to_floats(vec)
    if not vals:
        return [0.0] * target_dim
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size >= target_dim:
        return arr[:target_dim].tolist()
    pad = [0.0] * (target_dim - arr.size)
    return arr.tolist() + pad


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


def _resolve_credentials_path(project_id: str, credentials_path: Optional[str] = None) -> Optional[str]:
    if credentials_path:
        path = Path(credentials_path)
        if not path.is_file():
            raise FileNotFoundError(f"News credentials file not found: {path}")
        return str(path)

    for env_key in ("NEWS_BQ_CREDENTIALS", "NEWS_GOOGLE_APPLICATION_CREDENTIALS"):
        env_path = os.getenv(env_key)
        if env_path:
            path = Path(env_path)
            if path.is_file():
                return str(path)

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "airflow" / "big_query_test.json",
        Path("/data/ephemeral/home/pro-cv-finalproject-cv-07/airflow/big_query_test.json"),
        Path("/data/ephemeral/home/airflow/big_query_test.json"),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


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

    # Avoid BigQuery Storage API to prevent permission issues (readsessions.create).
    df = client.query(query).to_dataframe(create_bqstorage_client=False)

    if "trade_date" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"trade_date": "time"})

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df


def load_news_features_bq(
    *,
    project_id: str,
    dataset_id: str,
    table: str,
    commodity: str,
    key_word: Optional[str] = None,
    embedding_col: str = "news_embedding_mean",
    date_col: str = "collect_date",
    count_col: str = "collect_date_count",
    columns: Optional[Sequence[str]] = None,
    credentials_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load news features from BigQuery and normalize to a DataFrame suitable for TFT/CNN.
    Outputs columns:
      - collect_date (DATE)
      - date (YYYY-MM-DD string)
      - news_count (int)
      - embedding (JSON string)
      - news_emb_0..news_emb_511
      - other numeric stats (sentiment/timeframe) if present
    """
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError("google-cloud-bigquery is required for BigQuery access.") from exc

    if not key_word:
        key_word = COMMODITY_TO_KEYWORD.get(commodity.lower())
    if not key_word:
        raise ValueError(f"Unknown commodity for key_word mapping: {commodity}")

    select_cols = "*"
    if columns:
        select_cols = ", ".join(columns)

    query = (
        f"SELECT {select_cols} "
        f"FROM `{project_id}.{dataset_id}.{table}` "
        f"WHERE key_word = '{key_word}' "
        f"ORDER BY {date_col}"
    )

    cred_path = _resolve_credentials_path(project_id, credentials_path)
    if cred_path:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(cred_path)
        client = bigquery.Client(project=project_id, credentials=creds)
    else:
        client = bigquery.Client(project=project_id)
    # Avoid BigQuery Storage API to prevent permission issues (readsessions.create).
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        return df

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if embedding_col not in df.columns:
        raise ValueError(f"Missing embedding column: {embedding_col}")

    df = df.copy()
    df["collect_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df.dropna(subset=["collect_date"]).sort_values("collect_date")
    df["date"] = pd.to_datetime(df["collect_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if count_col in df.columns:
        df["news_count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)
    else:
        df["news_count"] = 0

    emb_list = [_ensure_embedding(vec, target_dim=512) for vec in df[embedding_col].tolist()]
    emb_matrix = pd.DataFrame(emb_list, columns=[f"news_emb_{i}" for i in range(512)])
    df["embedding"] = [json.dumps(vec) for vec in emb_list]

    drop_cols = {embedding_col, count_col, "key_word", "created_at"}
    if date_col != "collect_date":
        drop_cols.add(date_col)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Keep only numeric stats (besides embeddings/news_count).
    stat_cols = [
        c for c in df.columns
        if c not in {"collect_date", "date", "news_count", "embedding"}
        and not c.startswith("news_emb_")
    ]
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = pd.concat([df.reset_index(drop=True), emb_matrix], axis=1)
    base_cols = ["collect_date", "date", "news_count", "embedding"]
    emb_cols = [c for c in df.columns if c.startswith("news_emb_")]
    ordered_cols = base_cols + stat_cols + emb_cols
    return df[ordered_cols]
