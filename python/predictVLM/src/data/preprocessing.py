import ast
import numpy as np
import pandas as pd


def filtering_news(df):
    df = df.copy()
    
    df = df[df["filter_status"] == 'T']
    
    return df


def str_to_python(obj_str):
    if obj_str is None:
        return None
    return ast.literal_eval(str(obj_str).strip())


def handle_holidays(date_df, df):
    date_df = date_df.copy()
    df = df.copy()
    
    date_df['time'] = pd.to_datetime(date_df['time']).dt.date
    df['publish_date'] = pd.to_datetime(df['publish_date']).dt.date

    trade_days = date_df['time'].sort_values().unique()
    df['trade_date'] = df['publish_date'].apply(
        lambda d: trade_days[trade_days <= d][-1] if np.any(trade_days <= d) else pd.NaT
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    
    return df