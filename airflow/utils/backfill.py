import yfinance as yf
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import logging


# 순서대로 Cron, Wheat, Soybean, Gold, Sliver, Copper
SYMBOLS = ["ZC=F", "ZW=F", "ZS=F", "GC=F", "SI=F", "HG=F"]


load_dotenv("/data/ephemeral/home/airflow/.env")

project_id = os.getenv("BIGQUERY_PROJECT")
dataset_id = os.getenv("BIGQUERY_DATASET")
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
bucket_name = 'boostcamp-final-proj'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_yahoo_ohlcv(ticker, start, end):
    """
    Yahoo로부터 start일 ticker에 관한 OHLCV 추출
    """
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False, 
        progress=False
    )
    
    if df.empty:
        return df
    
    df = df.reset_index()
    df = df.rename(columns={
        "Date": "trade_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=True)
    
    df['trade_date'] = df['trade_date'].dt.date
    df["symbol"] = ticker
    df["ingest_time"] = datetime.utcnow().replace(tzinfo=timezone.utc)
    
    return df[["trade_date", "symbol", "open", "high", "low", "close", "ingest_time"]]


def load_ohlcv_to_bq(start_date, end_date):
    """
    ds일에 대한 OHLCV raw data를 BigQuery에 upload
    """
    client = bigquery.Client()
    
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = (datetime.strptime(end_date, "%Y-%m-%d").date()+ timedelta(days=1))
    
    df_list = []
    for sym in SYMBOLS:
        df = extract_yahoo_ohlcv(
            ticker=sym,
            start=start_date,
            end=end_date
        )
        if not df.empty:
            df_list.append(df)
    
    if not df_list:
        logger.info("수집한 데이터가 없습니다.")
    
    final_df = pd.concat(df_list, ignore_index=True)
    
    staging_table = f"{dataset_id}.raw_price_staging"
    target_table = f"{dataset_id}.raw_price"
    
    client.load_table_from_dataframe(
        final_df,
        staging_table,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE"
        )
    ).result()
    
    merge_sql = f"""
    MERGE `{target_table}` T
    USING `{staging_table}` S
    ON
      T.trade_date = S.trade_date
      AND T.symbol = S.symbol

    WHEN MATCHED THEN
      UPDATE SET
        open = S.open,
        high = S.high,
        low = S.low,
        close = S.close,
        ingest_time = S.ingest_time

    WHEN NOT MATCHED THEN
      INSERT (
        trade_date,
        symbol,
        open,
        high,
        low,
        close,
        ingest_time
      )
      VALUES (
        S.trade_date,
        S.symbol,
        S.open,
        S.high,
        S.low,
        S.close,
        S.ingest_time
      )
    """

    client.query(merge_sql).result()
    
    return final_df
    

def verify_data_quality(df, symbol, prev_close=None, change_threshold=0.1):
    """
    한 심볼에 대한 데이터 품질 검사 후 dictionary로 반환
    """
    log = {}
    log['trade_date'] = df['trade_date'].iloc[0]
    log['symbol'] = symbol
    
    # 결측치 체크
    log['NaN'] = df.isna().any().any() 
    
    # 타입 체크
    numeric_cols = ['open','high','low','close']
    log['type_error'] = not all(pd.api.types.is_numeric_dtype(df[col]) for col in numeric_cols)
    
    # 논리적 오류 체크
    open_vals = df['open'].values[0]
    high_vals = df['high'].values[0]
    low_vals = df['low'].values[0]
    close_vals = df['close'].values[0]
    
    logical_error = False
    if high_vals < low_vals or open_vals < 0 or close_vals < 0 or high_vals < 0 or low_vals < 0:
        logical_error = True
    log['logical_error'] = logical_error
    
    # 전일 대비 변화량 체크 -> 이상치 체크
    if prev_close is not None:
        change = abs(close_vals - prev_close)/prev_close
        log['outlier_change'] = change > change_threshold
    else:
        log['outlier_change'] = False
    
    return log


def check_stock_data(df):
    """
    raq 데이터의 품질을 검수하고, 로그 기록 후 BigQuery로 백업을 수행
    """
    client = bigquery.Client()
    
    logs = []
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        for i in range(1, len(group)):
            log = verify_data_quality(df=group.iloc[[i]], symbol=sym, prev_close=group["close"].iloc[i-1])
            logs.append(log)
    
    log_df = pd.DataFrame(logs)
    log_df["ingest_time"] = datetime.utcnow().replace(tzinfo=timezone.utc)

    staging_table = f"{dataset_id}.logs_staging"
    target_table = f"{dataset_id}.logs"
    
    client.load_table_from_dataframe(
        log_df,
        staging_table,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE"
        )
    ).result()
    
    merge_sql = f"""
    MERGE `{target_table}` T
    USING `{staging_table}` S
    ON
      T.trade_date = S.trade_date
      AND T.symbol = S.symbol

    WHEN MATCHED THEN
      UPDATE SET
        NaN = S.NaN,
        type_error = S.type_error,
        logical_error = S.logical_error,
        outlier_change = S.outlier_change,
        ingest_time = S.ingest_time

    WHEN NOT MATCHED THEN
      INSERT (
        trade_date,
        symbol,
        NaN,
        type_error,
        logical_error,
        outlier_change,
        ingest_time
      )
      VALUES (
        S.trade_date,
        S.symbol,
        S.NaN,
        S.type_error,
        S.logical_error,
        S.outlier_change,
        S.ingest_time
      )
    """
    client.query(merge_sql).result()


def daterange(start_date, end_date):
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)


if __name__ == "__main__":
    start_date = "2022-01-01"   # backfill 시작일
    end_date = "2025-12-31"   # backfill 종료일

    df = load_ohlcv_to_bq(start_date, end_date)
    check_stock_data(df)