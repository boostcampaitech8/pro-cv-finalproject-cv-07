#!/usr/bin/env python3
"""
뉴스 수집 및 저장 파이프라인
- 크롤링 -> T/F 판단 -> Entities/Triples 추출 -> Embedding -> VectorDB/BigQuery 저장

파이프라인 흐름:
1. 크롤링: Google News에서 뉴스 수집
2. T/F 판단: LLM으로 관련 뉴스 필터링
3. Entities/Triples 추출: (TODO) 지식 그래프 추출
4. Embedding 생성: article, entities, triples 임베딩
5. 저장: VectorDB + BigQuery
"""

import sys
from pathlib import Path

# 경로 설정 (상대 경로)
# main.py → news_preprocess → python → pro-cv-finalproject-cv-07
NEWS_PREPROCESS_ROOT = Path(__file__).resolve().parent  # news_preprocess/
PYTHON_ROOT = NEWS_PREPROCESS_ROOT.parent  # python/
REPO_ROOT = PYTHON_ROOT.parent  # pro-cv-finalproject-cv-07/

sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(NEWS_PREPROCESS_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
import os

# 내부 모듈
from engines.article_crawler import crawl_article, save_to_csv
from engines.status_judge import filter_relevant, unload_model
from engines.extract_kg import extract_kg_batch
from utils.bigquery_utils import save_articles, save_daily_summary
from utils.id_generator import add_ids_to_dataframes
from utils.graphdb_utils import save_to_graphdb, close_driver as close_graphdb

from engines.add_news_features import add_news_features
from utils.add_embedding import add_article_embedding
from utils.vectordb_utils import connect_vector_db, create_collection, push_data


def generate_embeddings(df: pd.DataFrame, entities_df: pd.DataFrame, triples_df: pd.DataFrame) -> dict:
    """
    Embedding 생성 (Bearer Token 방식)

    Args:
        df: 뉴스 DataFrame
        entities_df: 엔티티 DataFrame
        triples_df: 트리플 DataFrame

    Returns:
        dict with embeddings
    """
    print("\n[Embedding Generation]")

    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # Article embedding (title + description)
    article_texts = (
        df["title"].fillna("").astype(str)
        + "\n\n"
        + df["description"].fillna("").astype(str)
    ).tolist()

    print(f"  Generating article embeddings ({len(article_texts)} texts)...")
    article_embeddings = add_article_embedding(article_texts, access_key_id=access_key, secret_access_key=secret_key, dimensions=512)
    
    # Entity embeddings
    entity_texts = entities_df['entity_value'].tolist() if not entities_df.empty else []
    print(f"  Generating entity embeddings ({len(entity_texts)} texts)...")
    entity_embeddings = add_article_embedding(entity_texts, access_key_id=access_key, secret_access_key=secret_key, dimensions=1024) if entity_texts else []

    # Triple embeddings (value 컬럼 사용: "['subject', 'predicate', 'object']")
    triple_texts = triples_df['value'].tolist() if not triples_df.empty else []
    print(f"  Generating triple embeddings ({len(triple_texts)} texts)...")
    triple_embeddings = add_article_embedding(triple_texts, access_key_id=access_key, secret_access_key=secret_key, dimensions=1024) if triple_texts else []

    return {
        'article_embeddings': article_embeddings,
        'entity_embeddings': entity_embeddings,
        'triple_embeddings': triple_embeddings,
    }


# ============================================================
# VectorDB 저장
# ============================================================
def save_to_vectordb(df: pd.DataFrame):
    """
    VectorDB에 Article 임베딩 저장

    Args:
        df: 뉴스 DataFrame (id, embedding, collect_date, title, description 컬럼 필수)
    """
    print("\n[VectorDB Save]")

    if 'embedding' not in df.columns or df['embedding'].isna().all():
        print("  No embeddings to save")
        return

    # DB 연결
    client = connect_vector_db(os.getenv('VECTOR_DB_HOST'), os.getenv('VECTOR_DB_PORT'))

    # Collection 생성 (이미 존재하면 skip)
    try:
        create_collection(client, "news", dimension=512)
    except Exception:
        pass  # Collection already exists

    # push_data에 맞게 DataFrame 컬럼 매핑
    df_for_db = df[['id', 'embedding', 'collect_date', 'title', 'description', 'key_word']].copy()
    df_for_db = df_for_db.rename(columns={
        'embedding': 'article_embedding',
        'collect_date': 'trade_date',
        'key_word': 'type',
    })
    # trade_date가 datetime 객체여야 함 (push_data에서 strftime 호출)
    df_for_db['trade_date'] = pd.to_datetime(df_for_db['trade_date'])

    # 저장
    push_data(client, "news", df_for_db)
    print(f"  Articles: {len(df_for_db)} embeddings saved")
    


# ============================================================
# BigQuery 저장
# ============================================================
def save_to_bigquery(df: pd.DataFrame):
    """
    BigQuery에 Articles 데이터 저장

    Args:
        df: 뉴스 DataFrame
    """
    print("\n[BigQuery Save]")

    try:
        rows_saved = save_articles(df)
        print(f"  Total: {rows_saved} rows saved")
    except Exception as e:
        print(f"  BigQuery 저장 실패: {e}")
        print("  (BigQuery 설정을 확인하세요)")


# ============================================================
# 메인 파이프라인
# ============================================================
def run_pipeline(
    days_back: Optional[int] = None,
    start_hour: int = 0,
    start_minute: int = 0,
    end_date_str: Optional[str] = None,
    skip_tf_filter: bool = False,
    skip_embedding: bool = False,
    skip_features: bool = False,
    skip_vectordb: bool = False,
    skip_graphdb: bool = False,
    skip_bigquery: bool = False,
    save_csv: bool = True,
    model_name: Optional[str] = None,
    keep_model_loaded: bool = False,
) -> pd.DataFrame:
    """
    뉴스 수집 및 저장 파이프라인 실행

    Args:
        days_back: 수집 기간 (일)
        start_hour: 필터링 시작 시간 (0-23)
        start_minute: 필터링 시작 분 (0-59)
        end_date_str: 수집 종료 날짜 (YYYY-MM-DD), None이면 현재 시각 사용
        skip_tf_filter: T/F 필터링 건너뛰기
        skip_embedding: 임베딩 생성 건너뛰기
        skip_vectordb: VectorDB 저장 건너뛰기
        skip_graphdb: GraphDB 저장 건너뛰기
        skip_bigquery: BigQuery 저장 건너뛰기
        save_csv: CSV 파일 저장 여부
        model_name: T/F 판단에 사용할 모델명

    Returns:
        최종 처리된 DataFrame
    """
    print("=" * 60)
    print("News Collection & Save Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ========================================
    # Step 1: 크롤링
    # ========================================
    print("\n" + "=" * 40)
    print("Step 1: Crawling")
    print("=" * 40)

    df = crawl_article(days_back=days_back, start_hour=start_hour, start_minute=start_minute, end_date_str=end_date_str)

    if df.empty:
        print("No articles collected. Pipeline stopped.")
        return df

    print(f"\nCrawled {len(df)} articles")

    # ========================================
    # Step 2: T/F 판단
    # ========================================
    print("\n" + "=" * 40)
    print("Step 2: T/F Classification")
    print("=" * 40)

    if skip_tf_filter:
        print("  Skipped (skip_tf_filter=True)")
        relevant_df = df
    else:
        try:
            if model_name:
                relevant_df = filter_relevant(df, model_name=model_name)
            else:
                relevant_df = filter_relevant(df)
        except Exception as e:
            print(f"  T/F 판단 실패: {e}")
            print("  (모델 로딩 문제일 수 있음, 전체 데이터 사용)")
            relevant_df = df

    if relevant_df.empty:
        print("No relevant articles after filtering. Pipeline stopped.")
        return relevant_df

    print(f"\n{len(relevant_df)} relevant articles (T)")

    # index 정규화 (extract_kg_batch에서 article_id로 사용)
    relevant_df = relevant_df.reset_index(drop=True)

    # ========================================
    # Step 3: Entities/Triples 추출
    # ========================================
    print("\n" + "=" * 40)
    print("Step 3: Entities/Triples Extraction")
    print("=" * 40)

    entities_df, triples_df = extract_kg_batch(relevant_df, model_name=model_name)

    # T/F 판단 + KG 추출 완료 후 모델 해제
    if not keep_model_loaded:
        unload_model()

    print(f"\nExtracted {len(entities_df)} entities, {len(triples_df)} triples")

    # ========================================
    # Step 3.5: 고유 ID 생성
    # ========================================
    print("\n[ID Generation]")
    relevant_df, entities_df, triples_df = add_ids_to_dataframes(
        relevant_df, entities_df, triples_df
    )
    print(f"  Generated IDs for {len(relevant_df)} articles, {len(entities_df)} entities, {len(triples_df)} triples")

    # ========================================
    # Step 4: Embedding 생성
    # ========================================
    print("\n" + "=" * 40)
    print("Step 4: Embedding Generation")
    print("=" * 40)

    if skip_embedding:
        print("  Skipped (skip_embedding=True)")
        embeddings = None
    else:
        embeddings = generate_embeddings(relevant_df, entities_df, triples_df)
        relevant_df['embedding'] = embeddings['article_embeddings']
        entities_df['embedding'] = embeddings['entity_embeddings']
        triples_df['embedding'] = embeddings['triple_embeddings']

    # ========================================
    # Step 4.5: News Features (Sentiment + Timeframe)
    # ========================================
    print("\n" + "=" * 40)
    print("Step 4.5: News Features (Sentiment + Timeframe)")
    print("=" * 40)

    if skip_features:
        print("  Skipped (skip_features=True)")
    else:
        relevant_df = add_news_features(relevant_df)
        print(f"  Sentiment labels: {relevant_df['sentiment_label'].value_counts().to_dict()}")
        print(f"  Timeframe labels: {relevant_df['timeframe_label'].value_counts().to_dict()}")

    # ========================================
    # Step 5: Vector DB & GraphDB 저장
    # ========================================
    print("\n" + "=" * 40)
    print("Step 5: Embeddings DB Save")
    print("=" * 40)

    if skip_vectordb:
        print("  Skipped (skip_vectordb=True)")
    elif embeddings:
        save_to_vectordb(relevant_df)

    # GraphDB 저장 (entities, triples 포함)
    if skip_graphdb:
        print("  GraphDB: Skipped (skip_graphdb=True)")
    elif embeddings:
        try:
            save_to_graphdb(relevant_df, entities_df, triples_df)
        except Exception as e:
            print(f"  GraphDB 저장 실패: {e}")
            print("  (NEO4J_URI, NEO4J_PASSWORD 환경변수를 확인하세요)")
    # ========================================
    # Step 6: BigQuery 저장
    # ========================================
    print("\n" + "=" * 40)
    print("Step 6: BigQuery Save")
    print("=" * 40)

    if skip_bigquery:
        print("  Skipped (skip_bigquery=True)")
    else:
        save_to_bigquery(relevant_df)

        # Daily Summary 저장 (임베딩이 있을 때만)
        if embeddings and 'embedding' in relevant_df.columns:
            collect_date = relevant_df['collect_date'].iloc[0]
            collect_date_count = len(df)  # T 필터 전 전체 기사 수
            article_emb_array = np.array(embeddings['article_embeddings'])
            news_embedding_mean = article_emb_array.mean(axis=0).tolist()

            # sentiment / timeframe 집계 (feature_.py add_news_imformation_features 와 동일)
            news_features = None
            has_sentiment = 'sentiment_label' in relevant_df.columns
            has_timeframe = 'timeframe_label' in relevant_df.columns

            if has_sentiment or has_timeframe:
                news_features = {}

                if has_sentiment:
                    # score 통계: mean, std, max, min
                    news_features['sentiment_score_mean'] = float(relevant_df['sentiment_score'].mean())
                    news_features['sentiment_score_std'] = float(relevant_df['sentiment_score'].std())
                    news_features['sentiment_score_max'] = float(relevant_df['sentiment_score'].max())
                    news_features['sentiment_score_min'] = float(relevant_df['sentiment_score'].min())
                    # label 비율: normalize=True → 비율
                    sent_ratio = relevant_df['sentiment_label'].value_counts(normalize=True)
                    news_features['sentiment_neg_ratio'] = float(sent_ratio.get('negative', 0))
                    news_features['sentiment_neu_ratio'] = float(sent_ratio.get('neutral', 0))
                    news_features['sentiment_pos_ratio'] = float(sent_ratio.get('positive', 0))

                if has_timeframe:
                    news_features['timeframe_score_mean'] = float(relevant_df['timeframe_score'].mean())
                    news_features['timeframe_score_std'] = float(relevant_df['timeframe_score'].std())
                    news_features['timeframe_score_max'] = float(relevant_df['timeframe_score'].max())
                    news_features['timeframe_score_min'] = float(relevant_df['timeframe_score'].min())
                    tf_ratio = relevant_df['timeframe_label'].value_counts(normalize=True)
                    news_features['time_past_ratio'] = float(tf_ratio.get('past', 0))
                    news_features['time_present_ratio'] = float(tf_ratio.get('present', 0))
                    news_features['time_future_ratio'] = float(tf_ratio.get('future', 0))

            save_daily_summary(
                collect_date, collect_date_count, news_embedding_mean,
                news_features=news_features,
            )
            print(f"  Daily Summary: count={collect_date_count}, embedding_dim={len(news_embedding_mean)}")
            if news_features:
                print(f"  Features: {list(news_features.keys())}")

    # ========================================
    # CSV 저장 (선택)
    # ========================================
    if save_csv:
        print("\n" + "=" * 40)
        print("Saving CSV")
        print("=" * 40)
        save_to_csv(relevant_df)

    # ========================================
    # 완료
    # ========================================
    # 연결 정리
    try:
        close_graphdb()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("Pipeline Completed")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final articles: {len(relevant_df)}")

    return relevant_df

# airflow 수집 -> 전처리 -> 모델 학습 -> 저장
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뉴스 수집 및 저장 파이프라인")
    parser.add_argument("--days-back", type=int, default=1, help="수집 기간 (일)")
    parser.add_argument("--start-hour", type=int, default=13, help="필터링 시작 시간 (0-23)")
    parser.add_argument("--start-minute", type=int, default=45, help="필터링 시작 분 (0-59)")
    parser.add_argument("--end-date", type=str, default=None, help="수집 종료 날짜 (YYYY-MM-DD), 미지정 시 현재 시각")
    parser.add_argument("--skip-tf-filter", action="store_true", help="T/F 판단 건너뛰기")
    parser.add_argument("--skip-embedding", action="store_true", help="임베딩 생성 건너뛰기")
    parser.add_argument("--skip-features", action="store_true", help="Sentiment/Timeframe 피처 건너뛰기")
    parser.add_argument("--skip-vectordb", action="store_true", help="VectorDB 저장 건너뛰기")
    parser.add_argument("--skip-graphdb", action="store_true", help="GraphDB 저장 건너뛰기")
    parser.add_argument("--skip-bigquery", action="store_true", help="BigQuery 저장 건너뛰기")
    parser.add_argument("--save-csv", action="store_true", default=True, help="CSV 파일 저장")
    parser.add_argument("--no-save-csv", action="store_false", dest="save_csv", help="CSV 파일 저장 안함")
    parser.add_argument("--model-name", type=str, default=None, help="T/F 판단에 사용할 모델명")
    args = parser.parse_args()

    df = run_pipeline(
        days_back=args.days_back,
        start_hour=args.start_hour,
        start_minute=args.start_minute,
        end_date_str=args.end_date,
        skip_tf_filter=args.skip_tf_filter,
        skip_embedding=args.skip_embedding,
        skip_features=args.skip_features,
        skip_vectordb=args.skip_vectordb,
        skip_graphdb=args.skip_graphdb,
        skip_bigquery=args.skip_bigquery,
        save_csv=args.save_csv,
        model_name=args.model_name,
    )

    if not df.empty:
        print("\nSample data:")
        print(df[['title', 'key_word']].head(10))
