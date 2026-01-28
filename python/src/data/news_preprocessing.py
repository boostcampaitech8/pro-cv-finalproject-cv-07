import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ast


def parse_embedding(embedding_str: str) -> np.ndarray:
    """
    문자열로 된 embedding을 numpy array로 변환
    
    Args:
        embedding_str: "[0.1, 0.2, ...]" 형태의 문자열
        
    Returns:
        numpy array
    """
    try:
        if pd.isna(embedding_str):
            return None
        # 문자열을 리스트로 변환
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
    날짜별로 뉴스를 집계하여 embedding mean pooling과 기사 개수 계산
    
    Args:
        news_df: 뉴스 데이터프레임
        date_column: 날짜 컬럼명
        filter_column: 필터 상태 컬럼명
        embedding_column: embedding 컬럼명
        filter_value: 필터링할 값 (기본: 'T')
        
    Returns:
        날짜별 집계된 데이터프레임 (date, news_count, news_embedding_mean)
    """
    # 필터링
    filtered_df = news_df[news_df[filter_column] == filter_value].copy()
    
    # 날짜만 추출 (YYYY-MM-DD 형식)
    filtered_df['date'] = pd.to_datetime(filtered_df[date_column]).dt.date
    
    # embedding 파싱
    filtered_df['embedding_parsed'] = filtered_df[embedding_column].apply(parse_embedding)
    
    # None이 아닌 embedding만 남기기
    filtered_df = filtered_df[filtered_df['embedding_parsed'].notna()].copy()
    
    # 날짜별 그룹화
    grouped = filtered_df.groupby('date')
    
    # 결과 저장
    result_data = []
    
    for date, group in grouped:
        # 기사 개수
        news_count = len(group)
        
        # embedding mean pooling
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


def merge_news_with_price(
    price_df: pd.DataFrame,
    news_df: pd.DataFrame,
    date_column: str = 'time'
) -> pd.DataFrame:
    """
    가격 데이터와 뉴스 데이터를 병합
    
    Args:
        price_df: 가격 데이터프레임 (corn_future_price.csv)
        news_df: 집계된 뉴스 데이터프레임
        date_column: 가격 데이터의 날짜 컬럼명
        
    Returns:
        병합된 데이터프레임
    """
    # 가격 데이터 복사
    merged_df = price_df.copy()
    
    # 날짜 형식 통일
    merged_df['date'] = pd.to_datetime(merged_df[date_column]).dt.date
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # 뉴스 데이터와 병합
    merged_df = merged_df.merge(
        news_df[['date', 'news_count', 'news_embedding_mean']],
        on='date',
        how='left'
    )
    
    # 뉴스가 없는 날은 0으로 채우기
    merged_df['news_count'] = merged_df['news_count'].fillna(0).astype(int)
    
    # embedding이 없는 날은 zero vector로 채우기
    embedding_dim = news_df['news_embedding_mean'].iloc[0].shape[0] if len(news_df) > 0 else 768
    zero_embedding = np.zeros(embedding_dim, dtype=np.float32)
    
    merged_df['news_embedding_mean'] = merged_df['news_embedding_mean'].apply(
        lambda x: x if isinstance(x, np.ndarray) else zero_embedding
    )
    
    return merged_df


def expand_embedding_to_columns(
    df: pd.DataFrame,
    embedding_column: str = 'news_embedding_mean',
    prefix: str = 'news_emb'
) -> pd.DataFrame:
    """
    embedding을 개별 컬럼으로 확장
    
    Args:
        df: 데이터프레임
        embedding_column: embedding 컬럼명
        prefix: 새로운 컬럼의 접두사
        
    Returns:
        embedding이 확장된 데이터프레임
    """
    # embedding을 DataFrame으로 변환
    embedding_matrix = np.stack(df[embedding_column].values)
    embedding_df = pd.DataFrame(
        embedding_matrix,
        columns=[f'{prefix}_{i}' for i in range(embedding_matrix.shape[1])],
        index=df.index
    )
    
    # 원본 데이터프레임과 결합
    result_df = pd.concat([df.drop(columns=[embedding_column]), embedding_df], axis=1)
    
    return result_df


def create_multi_commodity_features(
    corn_df: pd.DataFrame,
    soybean_df: pd.DataFrame,
    wheat_df: pd.DataFrame,
    target_commodity: str = 'corn'
) -> pd.DataFrame:
    """
    여러 상품의 가격 정보를 하나의 데이터프레임으로 결합
    상관관계가 있는 다른 상품의 정보를 feature로 추가
    
    Args:
        corn_df: 옥수수 데이터
        soybean_df: 대두 데이터
        wheat_df: 밀 데이터
        target_commodity: 예측 대상 상품 ('corn', 'soybean', 'wheat')
        
    Returns:
        결합된 데이터프레임
    """
    # 날짜 컬럼 통일
    for df in [corn_df, soybean_df, wheat_df]:
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['time']).dt.date
            df['date'] = pd.to_datetime(df['date'])
    
    # 타겟 상품 선택
    commodity_map = {
        'corn': corn_df,
        'soybean': soybean_df,
        'wheat': wheat_df
    }
    
    target_df = commodity_map[target_commodity].copy()
    
    # 다른 상품들의 가격 정보 추가
    other_commodities = [k for k in commodity_map.keys() if k != target_commodity]
    
    for other in other_commodities:
        other_df = commodity_map[other]
        
        # OHLCV 컬럼만 선택하고 prefix 추가
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        existing_columns = [col for col in price_columns if col in other_df.columns]
        
        other_prices = other_df[['date'] + existing_columns].copy()
        other_prices.columns = ['date'] + [f'{other}_{col}' for col in existing_columns]
        
        # 병합
        target_df = target_df.merge(other_prices, on='date', how='left')
    
    return target_df


if __name__ == "__main__":
    # 예시 사용법
    import os
    
    # 데이터 경로
    data_dir = "./src/datasets"
    
    # 뉴스 데이터 로드
    news_df = pd.read_csv(os.path.join(data_dir, "news_articles_resources.csv"))
    
    # 가격 데이터 로드
    corn_df = pd.read_csv(os.path.join(data_dir, "corn_future_price.csv"))
    soybean_df = pd.read_csv(os.path.join(data_dir, "soybean_future_price.csv"))
    wheat_df = pd.read_csv(os.path.join(data_dir, "wheat_future_price.csv"))
    
    # 뉴스 집계
    print("뉴스 데이터 집계 중...")
    aggregated_news = aggregate_news_by_date(news_df)
    print(f"집계된 날짜 수: {len(aggregated_news)}")
    
    # 옥수수 가격과 병합
    print("\n가격 데이터와 병합 중...")
    corn_with_news = merge_news_with_price(corn_df, aggregated_news)
    
    # embedding을 개별 컬럼으로 확장
    print("Embedding 확장 중...")
    corn_expanded = expand_embedding_to_columns(corn_with_news)
    
    # 다른 상품 정보 추가 (옵션)
    print("\n다른 상품 정보 추가 중...")
    corn_multi = create_multi_commodity_features(
        corn_expanded, 
        soybean_df, 
        wheat_df, 
        target_commodity='corn'
    )
    
    # 결과 저장
    output_path = os.path.join(data_dir, "corn_with_news_features.csv")
    corn_multi.to_csv(output_path, index=False)
    print(f"\n전처리 완료! 저장 위치: {output_path}")
    print(f"최종 feature 개수: {len(corn_multi.columns)}")
