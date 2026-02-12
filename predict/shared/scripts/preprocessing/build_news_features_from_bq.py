#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _ensure_embedding(vec, target_dim: int = 512) -> np.ndarray:
    if vec is None or (isinstance(vec, float) and np.isnan(vec)):
        return np.zeros(target_dim, dtype=np.float32)
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size >= target_dim:
        return arr[:target_dim]
    pad = np.zeros(target_dim - arr.size, dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def _load_bq_table(
    project_id: str,
    dataset_id: str,
    table: str,
    key_word: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError("google-cloud-bigquery is required to load news data.") from exc

    client = bigquery.Client(project=project_id)
    select_cols = ", ".join(columns) if columns else "*"
    query = f"""
        SELECT {select_cols}
        FROM `{project_id}.{dataset_id}.{table}`
        WHERE key_word = '{key_word}'
        ORDER BY collect_date
    """
    return client.query(query).to_dataframe()


def build_news_features(
    df: pd.DataFrame,
    *,
    embedding_col: str = "news_embedding_mean",
    date_col: str = "collect_date",
    count_col: str = "collect_date_count",
) -> pd.DataFrame:
    if df.empty:
        return df

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if embedding_col not in df.columns:
        raise ValueError(f"Missing embedding column: {embedding_col}")

    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).sort_values("date")

    if count_col in df.columns:
        df["news_count"] = df[count_col].fillna(0).astype(int)
    else:
        df["news_count"] = 0

    # 중복 날짜가 있으면 마지막 값으로 정리
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    emb_list = [
        _ensure_embedding(vec, target_dim=512) for vec in df[embedding_col].tolist()
    ]
    emb_matrix = np.vstack(emb_list)

    emb_cols = [f"news_emb_{i}" for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)

    df["embedding"] = [json.dumps(vec.tolist()) for vec in emb_matrix]

    # 통계 컬럼 유지 (뉴스 관련 통계는 그대로 피처로 사용)
    drop_cols = [date_col, embedding_col]
    if count_col in df.columns:
        drop_cols.append(count_col)
    if "key_word" in df.columns:
        drop_cols.append("key_word")

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # 컬럼 순서 정리
    base_cols = ["date", "news_count", "embedding"]
    stat_cols = [c for c in df.columns if c not in base_cols + emb_cols]
    ordered_cols = base_cols + stat_cols + emb_cols
    return df[ordered_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--table", default="articles")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--commodities", nargs="+", required=True)
    parser.add_argument("--agri_keyword", default="agriculture")
    parser.add_argument("--metal_keyword", default="metal")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    group_map: Dict[str, str] = {}
    for c in args.commodities:
        if c.lower() in {"corn", "wheat", "soybean"}:
            group_map[c] = args.agri_keyword
        elif c.lower() in {"gold", "silver", "copper"}:
            group_map[c] = args.metal_keyword
        else:
            raise ValueError(f"Unknown commodity mapping: {c}")

    # key_word별로 한 번만 로드
    loaded: Dict[str, pd.DataFrame] = {}
    for key_word in set(group_map.values()):
        df = _load_bq_table(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            table=args.table,
            key_word=key_word,
            columns=[
                "collect_date",
                "collect_date_count",
                "news_embedding_mean",
                "sentiment_score_mean",
                "sentiment_score_std",
                "sentiment_score_max",
                "sentiment_score_min",
                "sentiment_neg_ratio",
                "sentiment_neu_ratio",
                "sentiment_pos_ratio",
                "timeframe_score_mean",
                "timeframe_score_std",
                "key_word",
            ],
        )
        loaded[key_word] = build_news_features(df)

    for commodity, key_word in group_map.items():
        out_dir = output_root / commodity
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "news_features.csv"
        df = loaded[key_word]
        if df.empty:
            print(f"[WARN] no rows for {commodity} (key_word={key_word})")
        df.to_csv(out_path, index=False)
        print(f"✓ saved: {out_path}")


if __name__ == "__main__":
    main()
