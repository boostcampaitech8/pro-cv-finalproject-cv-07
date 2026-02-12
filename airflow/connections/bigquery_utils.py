#!/usr/bin/env python3
"""
BigQuery 저장 유틸리티
- 뉴스 데이터를 BigQuery에 저장
- articles, entities, triples 테이블 분리
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Optional
from datetime import datetime, timezone

from google.cloud import bigquery
from google.oauth2 import service_account


# ============================================================
# BigQuery 설정
# ============================================================
PROJECT_ID = "gcp-practice-484218"
DATASET_ID = "news_data"
TABLE_ARTICLES = "articles"
TABLE_DAILY_SUMMARY = "daily_summary"


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # pro-cv-finalproject-cv-07/

def _get_client():
    KEY_PATH = os.getenv('BIG_QUERY_KEY_PATH', str(_REPO_ROOT / 'big_query_test.json'))
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    return bigquery.Client(credentials=credentials, project=PROJECT_ID)


def save_articles(df: pd.DataFrame, table_name: str = TABLE_ARTICLES) -> int:
    """
    뉴스 기사를 BigQuery에 저장

    Args:
        df: DataFrame (id, title, doc_url, description 등)
        table_name: 테이블명

    Returns:
        저장된 행 수
    """
    if df.empty:
        print("  No articles to save")
        return 0

    client = _get_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    # 저장할 컬럼
    save_columns = [
        'id', 'title', 'doc_url', 'description', 'all_text',
        'publish_date', 'meta_site_name', 'key_word','collect_date'
    ]

    existing_columns = [col for col in save_columns if col in df.columns]
    df_to_save = df[existing_columns].copy()
    df_to_save['created_at'] = datetime.now(timezone.utc)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="collect_date"
        ),
        clustering_fields=["key_word"]
    )

    try:
        job = client.load_table_from_dataframe(df_to_save, table_id, job_config=job_config)
        job.result()
        print(f"  Articles: {len(df_to_save)} rows -> {table_id}")
        return len(df_to_save)
    except Exception as e:
        print(f"  Articles Error: {e}")
        return 0


def save_article_daily_summary(collect_date, collect_date_count: int, news_embedding_mean: list,
                       news_features: Optional[dict] = None,
                       key_word: Optional[str] = None,
                       table_name: str = TABLE_DAILY_SUMMARY) -> int:
    """
    일별 요약 데이터를 BigQuery에 저장

    Args:
        collect_date: 수집 날짜 (date 객체)
        collect_date_count: 해당 날짜 기사 개수
        news_embedding_mean: 해당 날짜 article embedding 평균 (list)
        news_features: feature_.py add_news_imformation_features 와 동일한 컬럼 dict
            score 통계: sentiment_score_mean/std/max/min, timeframe_score_mean/std/max/min
            sentiment 비율: sentiment_neg_ratio, sentiment_neu_ratio, sentiment_pos_ratio
            timeframe 비율: time_past_ratio, time_present_ratio, time_future_ratio
        key_word: 요약 카테고리 (metal/agriculture)
        table_name: 테이블명

    Returns:
        저장된 행 수
    """
    client = _get_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    row = {
        'collect_date': collect_date,
        'collect_date_count': collect_date_count,
        'news_embedding_mean': news_embedding_mean,
        'created_at': datetime.now(timezone.utc),
    }
    if key_word is not None:
        row['key_word'] = str(key_word)
    if news_features is not None:
        row.update(news_features)

    df_to_save = pd.DataFrame([row])

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="collect_date"
        ),
    )

    try:
        job = client.load_table_from_dataframe(df_to_save, table_id, job_config=job_config)
        job.result()
        print(f"  Daily Summary: 1 row -> {table_id}")
        return 1
    except Exception as e:
        print(f"  Daily Summary Error: {e}")
        return 0


if __name__ == "__main__":
    print("BigQuery Utils - Test")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Tables: {TABLE_ARTICLES}, {TABLE_DAILY_SUMMARY}")
