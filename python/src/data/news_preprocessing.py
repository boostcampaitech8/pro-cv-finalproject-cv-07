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
    휴장일 뉴스는 바로 이전 거래일에 누적
    
    Args:
        price_df: 가격 데이터프레임
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
    
    # 뉴스 데이터 복사
    news_df = news_df.copy()
    
    # 거래일 목록 정렬 (가격 데이터에 있는 날짜들)
    trading_dates = sorted(merged_df['date'].dt.date)
    trading_dates_set = set(trading_dates)
    
    # 날짜별 뉴스 딕셔너리
    news_by_date = {}
    for _, row in news_df.iterrows():
        news_date = row['date'].date()
        news_by_date[news_date] = {
            'count': row['news_count'],
            'embedding': row['news_embedding_mean']
        }
    
    # 모든 뉴스 날짜 정렬
    all_news_dates = sorted(news_by_date.keys())
    
    # 거래일별로 뉴스 재배치
    accumulated_news = {}
    
    for news_date in all_news_dates:
        if news_date in trading_dates_set:
            # 거래일인 경우 - 그대로 저장
            if news_date not in accumulated_news:
                accumulated_news[news_date] = {
                    'counts': [],
                    'embeddings': []
                }
            accumulated_news[news_date]['counts'].append(news_by_date[news_date]['count'])
            accumulated_news[news_date]['embeddings'].append(news_by_date[news_date]['embedding'])
        else:
            # 휴장일인 경우 - 바로 이전 거래일 찾기
            prev_trading_date = None
            for td in reversed(trading_dates):
                if td < news_date:
                    prev_trading_date = td
                    break
            
            if prev_trading_date is not None:
                # 이전 거래일에 누적
                if prev_trading_date not in accumulated_news:
                    accumulated_news[prev_trading_date] = {
                        'counts': [],
                        'embeddings': []
                    }
                accumulated_news[prev_trading_date]['counts'].append(news_by_date[news_date]['count'])
                accumulated_news[prev_trading_date]['embeddings'].append(news_by_date[news_date]['embedding'])
    
    # 누적된 뉴스를 평균내기
    final_news = []
    for trade_date, data in accumulated_news.items():
        combined_count = sum(data['counts'])
        combined_embedding = np.mean(data['embeddings'], axis=0)
        
        final_news.append({
            'date': pd.Timestamp(trade_date),
            'news_count': combined_count,
            'news_embedding_mean': combined_embedding
        })
    
    final_news_df = pd.DataFrame(final_news)
    
    # 가격 데이터와 병합
    merged_df = merged_df.merge(
        final_news_df[['date', 'news_count', 'news_embedding_mean']],
        on='date',
        how='left'
    )
    
    # 뉴스가 없는 날은 0으로 채우기
    merged_df['news_count'] = merged_df['news_count'].fillna(0).astype(int)
    
    # embedding이 없는 날은 zero vector로 채우기
    embedding_dim = 512
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
    
    result_df = df.copy()
    
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
    import os
    
    # 데이터 경로
    data_dir = "../datasets"
    
    print("="*60)
    print("뉴스 데이터 전처리 시작")
    print("="*60)
    
    # 뉴스 데이터 로드
    print("\n1. 뉴스 데이터 로드 중...")
    news_df = pd.read_csv(os.path.join(data_dir, "news_articles_resources.csv"))
    print(f"   ✓ 뉴스 기사 수: {len(news_df)}")
    
    # 뉴스 집계
    print("\n2. 날짜별 뉴스 집계 중...")
    aggregated_news = aggregate_news_by_date(news_df)
    print(f"   ✓ 집계된 날짜 수: {len(aggregated_news)}")
    
    # 각 상품별로 처리
    commodities = ['corn', 'soybean', 'wheat']
    
    for commodity in commodities:
        print(f"\n{'='*60}")
        print(f"{commodity.upper()} 처리 중...")
        print(f"{'='*60}")
        
        # feature_engineering.csv 파일 로드 (이게 중요!)
        input_file = f"{commodity}_feature_engineering.csv"
        input_path = os.path.join(data_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"   ⚠️  경고: {input_file} 파일이 없습니다. 건너뜁니다.")
            continue
        
        print(f"\n   1) {input_file} 로드 중...")
        price_df = pd.read_csv(input_path)
        print(f"      ✓ 데이터 행 수: {len(price_df)}")
        print(f"      ✓ 기존 컬럼 수: {len(price_df.columns)}")
        
        # 가격 데이터와 뉴스 병합
        print(f"\n   2) 뉴스 데이터와 병합 중...")
        merged_df = merge_news_with_price(price_df, aggregated_news)
        print(f"      ✓ 병합 완료")
        
        # Embedding을 개별 컬럼으로 확장
        print(f"\n   3) Embedding 확장 중...")
        expanded_df = expand_embedding_to_columns(merged_df)
        print(f"      ✓ 확장된 컬럼 수: {len(expanded_df.columns)}")
        
        # 결과 저장
        output_file = f"{commodity}_with_news_features.csv"
        output_path = os.path.join(data_dir, output_file)
        expanded_df.to_csv(output_path, index=False)
        
        print(f"\n   ✅ 완료: {output_file}")
        print(f"      - 저장 위치: {output_path}")
        print(f"      - 최종 행 수: {len(expanded_df)}")
        print(f"      - 최종 컬럼 수: {len(expanded_df.columns)}")
    
    print(f"\n{'='*60}")
    print("✅ 모든 전처리 완료!")
    print(f"{'='*60}\n")
    
    print("생성된 파일:")
    for commodity in commodities:
        output_file = f"{commodity}_with_news_features.csv"
        output_path = os.path.join(data_dir, output_file)
        if os.path.exists(output_path):
            print(f"  ✓ {output_file}")
        else:
            print(f"  ✗ {output_file} (생성 실패)")
