import numpy as np
import pandas as pd


def add_log_return_feature(df):
    df = df.copy()
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    return df


def add_ema_features(df, spans=[5, 10, 20, 50, 100]):
    df = df.copy()
    
    for span in spans:
        df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    
    return df


def add_volatility_features(df, windows=[7, 14, 21]):
    df = df.copy()
    
    for window in windows:
        df[f'vol_return_{window}d'] = df['log_return'].rolling(window=window).std()
        df[f'vol_volume_{window}d'] = df['Volume'].rolling(window=window).std()
    
    return df


def add_news_count_feature(df, news_df):
    df = df.copy()
    news_df = news_df.copy()
    
    news_df['publish_date'] = pd.to_datetime(news_df['publish_date']).dt.date
    df['time'] = pd.to_datetime(df['time']).dt.date
    
    # 휴장일 기사 -> 다음 거래일로 매핑
    trade_days = df['time'].sort_values().unique()
    news_df['trade_date'] = news_df['publish_date'].apply(
        lambda d: trade_days[trade_days >= d][0] if np.any(trade_days >= d) else pd.NaT
    )
    
    daily_news_count = news_df.groupby('trade_date').size().reset_index(name='news_count')

    df = df.merge(daily_news_count, left_on='time', right_on='trade_date', how='left')
    df['news_count'] = df['news_count'].fillna(0)
    
    df.drop(columns=['trade_date'], inplace=True)
    return df


def add_news_imformation_features(df, news_df):
    df = df.copy()
    news_df = news_df.copy()
    
    news_df['publish_date'] = pd.to_datetime(news_df['publish_date']).dt.date
    df['time'] = pd.to_datetime(df['time']).dt.date

    trade_days = df['time'].sort_values().unique()
    news_df['trade_date'] = news_df['publish_date'].apply(
        lambda d: trade_days[trade_days >= d][0] if np.any(trade_days >= d) else pd.NaT
    )
    
    sentiment_counts = news_df.groupby('trade_date')['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0)
    sentiment_counts.columns = ['sentiment_neg_ratio', 'sentiment_neu_ratio', 'sentiment_pos_ratio']

    timeframe_counts = news_df.groupby('trade_date')['timeframe_label'].value_counts(normalize=True).unstack(fill_value=0)
    timeframe_counts.columns = ['time_past_ratio', 'time_present_ratio', 'time_future_ratio']

    score_stats = news_df.groupby('trade_date').agg({
        'sentiment_score': ['mean', 'std', 'max', 'min'],
        'timeframe_score': ['mean', 'std', 'max', 'min']
    })
    score_stats.columns = ['sentiment_score_mean', 'sentiment_score_std', 'sentiment_score_max', 'sentiment_score_min',
                           'timeframe_score_mean', 'timeframe_score_std', 'timeframe_score_max', 'timeframe_score_min']

    daily_df = pd.concat([score_stats, sentiment_counts, timeframe_counts], axis=1).reset_index()

    df = df.merge(daily_df, left_on='time', right_on='trade_date', how='left')
    
    df.drop(columns=['trade_date'], inplace=True)
    return df