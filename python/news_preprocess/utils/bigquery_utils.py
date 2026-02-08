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
TABLE_ENTITIES = "entities"
TABLE_TRIPLES = "triples"
TABLE_DAILY_SUMMARY = "daily_summary"


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # pro-cv-finalproject-cv-07/

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
        'publish_date', 'meta_site_name', 'key_word', 'embedding', 'collect_date'
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


def save_entities(entities_df: pd.DataFrame, table_name: str = TABLE_ENTITIES) -> int:
    """
    엔티티를 BigQuery에 저장

    Args:
        entities_df: DataFrame (id, entity_value, article_uuid)
        table_name: 테이블명

    Returns:
        저장된 행 수
    """
    if entities_df.empty:
        print("  No entities to save")
        return 0

    client = _get_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    # 저장할 컬럼
    save_columns = ['id', 'entity_value', 'article_uuid', 'embedding']

    existing_columns = [col for col in save_columns if col in entities_df.columns]
    df_to_save = entities_df[existing_columns].copy()
    df_to_save['created_at'] = datetime.now(timezone.utc)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
    )

    try:
        job = client.load_table_from_dataframe(df_to_save, table_id, job_config=job_config)
        job.result()
        print(f"  Entities: {len(df_to_save)} rows -> {table_id}")
        return len(df_to_save)
    except Exception as e:
        print(f"  Entities Error: {e}")
        return 0


def save_triples(triples_df: pd.DataFrame, table_name: str = TABLE_TRIPLES) -> int:
    """
    트리플(관계)을 BigQuery에 저장

    Args:
        triples_df: DataFrame (id, value, article_uuid)
        table_name: 테이블명

    Returns:
        저장된 행 수
    """
    if triples_df.empty:
        print("  No triples to save")
        return 0

    client = _get_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    # 저장할 컬럼 (value: "['subject', 'predicate', 'object']" 형태)
    save_columns = ['id', 'value', 'article_uuid', 'embedding']

    existing_columns = [col for col in save_columns if col in triples_df.columns]
    df_to_save = triples_df[existing_columns].copy()
    df_to_save['created_at'] = datetime.now(timezone.utc)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
    )

    try:
        job = client.load_table_from_dataframe(df_to_save, table_id, job_config=job_config)
        job.result()
        print(f"  Triples: {len(df_to_save)} rows -> {table_id}")
        return len(df_to_save)
    except Exception as e:
        print(f"  Triples Error: {e}")
        return 0


def save_daily_summary(collect_date, collect_date_count: int, news_embedding_mean: list,
                       news_features: Optional[dict] = None,
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
    if news_features is not None:
        row.update(news_features)

    df_to_save = pd.DataFrame([row])

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
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


def save_all(
    df: pd.DataFrame,
    entities_df: pd.DataFrame,
    triples_df: pd.DataFrame
) -> dict:
    """
    Articles, Entities, Triples 모두 저장

    Args:
        df: 뉴스 DataFrame
        entities_df: 엔티티 DataFrame
        triples_df: 트리플 DataFrame

    Returns:
        저장 결과 dict
    """
    results = {
        'articles': save_articles(df),
        'entities': save_entities(entities_df),
        'triples': save_triples(triples_df),
    }
    return results


def query_articles(keyword: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
    """
    BigQuery에서 기사 조회
    """
    client = _get_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ARTICLES}"

    if keyword:
        query = f"""
        SELECT * FROM `{table_id}`
        WHERE key_word = @keyword
        ORDER BY created_at DESC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("keyword", "STRING", keyword),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
    else:
        query = f"""
        SELECT * FROM `{table_id}`
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        job_config = None

    return client.query(query, job_config=job_config).to_dataframe()


def query_article_with_kg(article_id: str) -> dict:
    """
    특정 기사와 관련된 entities, triples 조회 (JOIN)

    Args:
        article_id: 기사 UUID

    Returns:
        dict with article, entities, triples
    """
    client = _get_client()

    # Article
    article_query = f"""
    SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ARTICLES}`
    WHERE id = @article_id
    """

    # Entities
    entities_query = f"""
    SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ENTITIES}`
    WHERE article_uuid = @article_id
    """

    # Triples
    triples_query = f"""
    SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_TRIPLES}`
    WHERE article_uuid = @article_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("article_id", "STRING", article_id),
        ]
    )

    article_df = client.query(article_query, job_config=job_config).to_dataframe()
    entities_df = client.query(entities_query, job_config=job_config).to_dataframe()
    triples_df = client.query(triples_query, job_config=job_config).to_dataframe()

    return {
        'article': article_df,
        'entities': entities_df,
        'triples': triples_df,
    }


if __name__ == "__main__":
    print("BigQuery Utils - Test")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Tables: {TABLE_ARTICLES}, {TABLE_ENTITIES}, {TABLE_TRIPLES}")