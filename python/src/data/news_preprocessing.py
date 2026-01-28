import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ast
import os


def parse_embedding(embedding_str: str) -> np.ndarray:
    """
    문자열로 된 embedding을 numpy array로 변환
    """
    try:
        if pd.isna(embedding_str):
            return None
        embedding_list = ast.literal_eval(embedding_str)
        return np.array(embedding_list, dtype=np.float32)
    except:
        return None


def aggregate_news_by_date(
    news_df: pd.DataFrame,
    date_column: str = 'publish_date',
    filter_column: str = 'filter_status',
    embedding_column: str = 'article_embedding',
    filter_value: str = 'T'
) -> pd.DataFrame:
    """
    날짜별로 뉴스를 집계 (상품 무관, 모든 날짜)
    """
    # 필터링
    filtered_df = news_df[news_df[filter_column] == filter_value].copy()
    
    # 날짜만 추출
    filtered_df['date'] = pd.to_datetime(filtered_df[date_column]).dt.date
    
    # embedding 파싱
    filtered_df['embedding_parsed'] = filtered_df[embedding_column].apply(parse_embedding)
    
    # None이 아닌 embedding만 남기기
    filtered_df = filtered_df[filtered_df['embedding_parsed'].notna()].copy()
    
    # 날짜별 그룹화
    grouped = filtered_df.groupby('date')
    
    result_data = []
    
    for date, group in grouped:
        news_count = len(group)
        embeddings = np.stack(group['embedding_parsed'].values)
        news_embedding_mean = embeddings.mean(axis=0)
        
        result_data.append({
            'date': date,
            'news_count': news_count,
            'news_embedding_mean': news_embedding_mean
        })
    
    result_df = pd.DataFrame(result_data)
    result_df['date'] = pd.to_datetime(result_df['date'])
    
    return result_df


def expand_embedding_to_columns(
    df: pd.DataFrame,
    embedding_column: str = 'news_embedding_mean',
    prefix: str = 'news_emb'
) -> pd.DataFrame:
    """
    embedding을 개별 컬럼으로 확장
    """
    embedding_matrix = np.stack(df[embedding_column].values)
    embedding_df = pd.DataFrame(
        embedding_matrix,
        columns=[f'{prefix}_{i}' for i in range(embedding_matrix.shape[1])],
        index=df.index
    )
    
    result_df = pd.concat([
        df.drop(columns=[embedding_column]),
        embedding_df
    ], axis=1)
    
    return result_df


if __name__ == "__main__":
    
    data_dir = "../datasets"
    
    print("="*60)
    print("뉴스 데이터 전처리 (통합 파일 생성)")
    print("="*60)
    
    # 뉴스 데이터 로드
    print("\n1. 뉴스 데이터 로드 중...")
    news_path = os.path.join(data_dir, "news_articles_resources.csv")
    
    if not os.path.exists(news_path):
        print(f"❌ Error: {news_path}")
        exit(1)
    
    news_df = pd.read_csv(news_path)
    print(f"   ✓ 뉴스 기사 수: {len(news_df)}")
    
    # 날짜별 집계
    print("\n2. 날짜별 뉴스 집계 중...")
    aggregated_news = aggregate_news_by_date(news_df)
    print(f"   ✓ 집계된 날짜 수: {len(aggregated_news)}")
    print(f"   ✓ 날짜 범위: {aggregated_news['date'].min().date()} ~ {aggregated_news['date'].max().date()}")
    
    # Embedding 확장
    print("\n3. Embedding 확장 중...")
    expanded_news = expand_embedding_to_columns(aggregated_news)
    print(f"   ✓ 총 컬럼 수: {len(expanded_news.columns)}")
    print(f"      - date: 1")
    print(f"      - news_count: 1")
    print(f"      - news_emb_0 ~ news_emb_511: 512")
    
    # 저장
    output_file = "news_features.csv"
    output_path = os.path.join(data_dir, output_file)
    expanded_news.to_csv(output_path, index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print("✅ 완료!")
    print(f"{'='*60}")
    print(f"\n파일: {output_path}")
    print(f"  - 행 수: {len(expanded_news)}")
    print(f"  - 컬럼 수: {len(expanded_news.columns)}")
    print(f"  - 파일 크기: {file_size_mb:.2f} MB")
    
    print(f"\n{'='*60}")
    print("이 파일은 모든 상품에서 공유됩니다:")
    print("="*60)
    print("""
생성된 파일: news_features.csv
  - 모든 날짜의 뉴스 포함 (거래일 + 휴장일)
  - 각 상품 로딩 시 해당 거래일에 맞춰 자동 정렬
  
사용 예시:
  corn_future_price.csv + news_features.csv
    → corn 거래일에 맞춰 merge
    → 휴장일 뉴스는 자동으로 이전 거래일과 통합
    
장점:
  ✓ 뉴스 파일 하나만 (용량 최소화)
  ✓ 모든 상품에서 공유
  ✓ 새 상품 추가 용이
    """)