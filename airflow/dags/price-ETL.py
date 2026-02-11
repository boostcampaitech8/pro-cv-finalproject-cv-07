import yfinance as yf

import pandas as pd
import numpy as np

from io import BytesIO
from google.cloud import bigquery, storage
import mysql.connector
from sqlalchemy import text
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime, timedelta, timezone
import urllib

import time
import mplfinance as mpf
import matplotlib.pyplot as plt
from PIL import Image

import os
from dotenv import load_dotenv
import logging


# 순서대로 Cron, Wheat, Soybean, Gold, Sliver, Copper
SYMBOLS = ["ZC=F", "ZW=F", "ZS=F", "GC=F", "SI=F", "HG=F"]

logger = logging.getLogger("airflow.task")

_env_candidates = [
    os.getenv("AIRFLOW_ENV"),
    os.getenv("AIRFLOW_HOME") and os.path.join(os.getenv("AIRFLOW_HOME"), ".env"),
    "/data/ephemeral/home/pro-cv-finalproject-cv-07/airflow/.env",
    "/data/ephemeral/home/airflow/.env",
]
for _env_path in _env_candidates:
    if _env_path and os.path.exists(_env_path):
        load_dotenv(_env_path)
        break

project_id = os.getenv("BIGQUERY_PROJECT")
dataset_id = os.getenv("BIGQUERY_DATASET")
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
bucket_name = 'boostcamp-final-proj'

if credentials_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
else:
    logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. BQ/GCS access may fail.")


def extract_yahoo_ohlcv(ticker, start, end):
    """
    Yahoo로부터 start일 ticker에 관한 OHLCV 추출
    """
    df = yf.download(
        ticker,
        start=(start - timedelta(days=1)).isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
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
    
    df = df[df["trade_date"] == start.isoformat()]
    
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index(drop=True)
    
    df['trade_date'] = df['trade_date'].dt.date
    df["symbol"] = ticker
    df["ingest_time"] = datetime.utcnow().replace(tzinfo=timezone.utc)
    
    return df[["trade_date", "symbol", "open", "high", "low", "close", "ingest_time"]]


def load_ohlcv(ds):
    """
    ds일에 대한 OHLCV raw data를 BigQuery와 Google Cloud SQL에 upload
    """
    client = bigquery.Client()
    
    yesterday = datetime.strptime(ds, "%Y-%m-%d").date()
    today = (yesterday + timedelta(days=1))
    
    df_list = []
    for sym in SYMBOLS:
        df = extract_yahoo_ohlcv(ticker=sym, start=yesterday, end=today)
        if not df.empty:
            df_list.append(df)
    
    if not df_list:
        logger.info(f"{yesterday}에 수집할 데이터가 없습니다.")
        raise AirflowSkipException(f"데이터 수집 X ({yesterday}) -> 뒤 작업 skip")
    
    final_df = pd.concat(df_list, ignore_index=True)
    
    staging_table = f"{dataset_id}.raw_price_staging"
    target_table = f"{dataset_id}.raw_price"
    
    client.load_table_from_dataframe(
        final_df,
        staging_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    ).result()
    
    merge_sql = f"""
    MERGE `{target_table}` T
    USING `{staging_table}` S
    ON T.trade_date = S.trade_date AND T.symbol = S.symbol
    WHEN MATCHED THEN
      UPDATE SET open=S.open, high=S.high, low=S.low, close=S.close, ingest_time=S.ingest_time
    WHEN NOT MATCHED THEN
      INSERT (trade_date, symbol, open, high, low, close, ingest_time)
      VALUES (S.trade_date, S.symbol, S.open, S.high, S.low, S.close, S.ingest_time)
    """
    client.query(merge_sql).result()
    logger.info("BigQuery Raw 데이터 업데이트 완료")

    transform_sql = f"""
    WITH calc AS (
        SELECT 
            symbol,
            trade_date,
            close,
            LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) as prev_close
        FROM `{target_table}`
    )
    SELECT 
        symbol,
        trade_date as base_date, 
        close as closing_price, 
        prev_close as prev_closing_price,
        (close - prev_close) as price_change,
        ((close - prev_close) / NULLIF(prev_close, 0)) * 100 as change_rate
    FROM calc
    WHERE trade_date = '{ds}'
    """
    transformed_df = client.query(transform_sql).to_dataframe()

    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST')
    db = os.getenv('DB_NAME')

    conn = None
    try:
        logger.info(f"[{host}] 순수 드라이버로 MySQL 연결 시도...")
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=db,
            connect_timeout=10
        )
        
        query = "SELECT product_id, ticker FROM product"
        product_map = pd.read_sql(query, con=conn)
        
        upload_df = transformed_df.merge(product_map, left_on='symbol', right_on='ticker')
        
        cursor = conn.cursor()
        insert_sql = """
        INSERT INTO daily_price 
        (product_id, base_date, closing_price, prev_closing_price, price_change, change_rate)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        data_to_insert = [
            (
                int(row['product_id']),
                row['base_date'],
                float(row['closing_price']),
                float(row['prev_closing_price']) if pd.notnull(row['prev_closing_price']) else 0.0,
                float(row['price_change']) if pd.notnull(row['price_change']) else 0.0,
                float(row['change_rate']) if pd.notnull(row['change_rate']) else 0.0
            ) 
            for _, row in upload_df.iterrows()
        ]

        if data_to_insert:
            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            logger.info(f"✅ Cloud SQL 적재 성공: {len(data_to_insert)}건")
        else:
            logger.warning("적재할 데이터가 매핑되지 않았습니다 (Ticker 불일치 확인 필요).")
        
    except Exception as e:
        logger.error(f"❌ MySQL 적재 중 에러 발생: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            logger.info("DB 연결 종료.")


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


def check_stock_data(ds):
    """
    raq 데이터의 품질을 검수하고, 로그 기록 후 BigQuery로 백업을 수행
    """
    client = bigquery.Client()
    
    yesterday = datetime.strptime(ds, "%Y-%m-%d").date()

    query = f"""
    SELECT
      trade_date,
      symbol,
      open,
      high,
      low,
      close,
    FROM `{dataset_id}.raw_price`
    WHERE symbol IN UNNEST({SYMBOLS})
      AND trade_date BETWEEN DATE_SUB('{yesterday}', INTERVAL 1 DAY)
                          AND '{yesterday}'
    """
    
    price_df = client.query(query).to_dataframe()

    if price_df.empty:
        logger.info(f"{yesterday} 데이터 없음")
        return

    logs = []

    for sym, group in price_df.groupby("symbol"):
        today_df = group[group["trade_date"] == yesterday]
        if today_df.empty:
            logger.info(f"{sym} - {yesterday} 데이터 없음")
            continue

        prev_df = group[group["trade_date"] == yesterday - timedelta(days=1)]
        prev_close = prev_df["close"].iloc[0] if not prev_df.empty else None

        log_entry = verify_data_quality(
            df=today_df.reset_index(drop=True),
            symbol=sym,
            prev_close=prev_close
        )

        logs.append(log_entry)
        logger.info(f"Checked {sym} for {yesterday}: {log_entry}")
    
    if not logs:
        logger.info(f"No logs to upload for {yesterday}")
        return

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


def generate_stock_features(df):
    """
    OHLCV raw data를 바탕으로 모델 학습에 필요한 피쳐 생성
    """
    df = df.copy()
    df = df.sort_values("trade_date").reset_index(drop=True)
    
    # 사용할 price 관련 모든 feature 제작
    horizons=[i for i in range(1, 21)]
    spans=[5, 10, 20, 50, 100, 200]
    windows=[7, 14, 21]
    
    df['close'] = df['close'].astype('float64')
    
    for span in spans:
        df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False, min_periods=span).mean()

    df['return_lag_1'] = np.log(df['close'].shift(1) / df['close'].shift(2))
    
    for window in windows:
        df[f'vol_return_{window}d'] = df['return_lag_1'].rolling(window=window, min_periods=window).std()
    
    df.drop(columns=['return_lag_1'], inplace=True)
    
    train_df = df.copy()
        
    for h in horizons:
        train_df[f'log_return_{h}'] = np.log(train_df['close'].shift(-(h-1)) / df['close'].shift(1))

    train_df["ingest_time"] = datetime.utcnow().replace(tzinfo=timezone.utc)
    df["ingest_time"] = datetime.utcnow().replace(tzinfo=timezone.utc)
    
    train_df = train_df.dropna()
    df = df.dropna()
    
    return train_df, df.iloc[-(max(horizons) - 1):]


def split_and_store_datasets(ds):
    """
    생성된 피쳐 데이터를 학습 및 추론 용으로 분리하여 BigQuery에 저장
    """
    client = bigquery.Client()
    
    yesterday = datetime.strptime(ds, "%Y-%m-%d").date()
    
    query = f"""
    SELECT *
    FROM `{dataset_id}.raw_price`
    WHERE symbol IN UNNEST({SYMBOLS})
      AND trade_date <= '{yesterday}'
    ORDER BY symbol, trade_date
    """
    price_df = client.query(query).to_dataframe()
    
    if price_df.empty:
        logger.info(f"{yesterday} 이전 데이터가 없습니다.")
        return
    
    train_dfs = []
    inference_dfs = []
    
    for sym, group in price_df.groupby("symbol"):
        train_df, inference_df = generate_stock_features(group.reset_index(drop=True))
        train_dfs.append(train_df)
        inference_dfs.append(inference_df) 
    
    final_train = pd.concat(train_dfs, ignore_index=True)
    final_inference = pd.concat(inference_dfs, ignore_index=True)
    
    if not final_train.empty:
        cutoff_date = yesterday - pd.Timedelta(days=3*365 + 90)
        final_train = final_train[final_train['trade_date'] >= cutoff_date]

    if not final_train.empty:
        client.load_table_from_dataframe(
            final_train,
            f"{dataset_id}.train_price",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        ).result()
    
    if not final_inference.empty:
        client.load_table_from_dataframe(
            final_inference,
            f"{dataset_id}.inference_price",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        ).result()


def load_all_data(date):
    """
    차트 이미지 생성을 위해 전체 데이터 불러오기 (train + inference)
    """
    client = bigquery.Client()
    cols = [
        "trade_date", "symbol", "open", "high", "low", "close",
        "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_100", "EMA_200"
    ]
    
    query_train = f"""
        SELECT *
        FROM `{dataset_id}.train_price`
        WHERE symbol IN UNNEST({SYMBOLS})
          AND trade_date <= '{date}'
    """
    df_train = client.query(query_train).to_dataframe()

    query_infer = f"""
        SELECT *
        FROM `{dataset_id}.inference_price`
        WHERE symbol IN UNNEST({SYMBOLS})
          AND trade_date <= '{date}'
    """
    df_infer = client.query(query_infer).to_dataframe()

    df = pd.concat([df_train, df_infer], ignore_index=True)
    df = df[cols]  
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values("trade_date").reset_index(drop=True)
    df = df.set_index('trade_date')
    
    return df


def get_existing_blobs(bucket, prefix):
    """
    prefix 기준으로 존재하는 모든 blob 이름 가져오기
    """
    return set(blob.name for blob in bucket.list_blobs(prefix=prefix))


def generate_and_upload_chart(
    df: pd.DataFrame,
    windows: list,
    emas_list: list,  # list of lists, e.g., [[], [20], [5, 10, 20]]
    end_start: str,
    end_stop: str,
    bucket_name: str,
    symbol: str,
    chart_type: str,
    image_size: int
):
    """
    주가 데이터를 시각화하여 차트 이미지를 생성하고 Google Cloud Storage에 업로드
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    dpi = plt.rcParams["figure.dpi"]
    figsize = (504 / dpi, 563 / dpi)

    end_start_ts = pd.Timestamp(end_start)
    end_stop_ts = pd.Timestamp(end_stop)
    ends = df.index[(df.index <= end_start_ts) & (df.index >= end_stop_ts)]
    pos_arr = df.index.get_indexer(ends)

    print(f"Date range: {end_stop} ~ {end_start}")
    print(f"Total target dates: {len(ends)}")
    
    existing_blobs = get_existing_blobs(bucket, prefix=f"{symbol}/")

    for w in windows:
        for emas in emas_list:
            saved = 0
            skipped = 0
            t0 = time.time()
            
            ema_suffix = "ema0" if len(emas) == 0 else "ema" + "_".join(map(str, emas))

            # EMA 컬럼 검증
            for ema in emas:
                ema_col = f"EMA_{ema}"
                if ema_col not in df.columns:
                    raise KeyError(f"EMA_{ema} column not found. Available columns: {list(df.columns)}")

            print(f"\nGenerating: window={w}, emas={emas if emas else 'None'}")

            for end, pos in zip(ends, pos_arr):
                start = pos - w
                if start < 0:
                    skipped += 1
                    continue

                df_win = df.iloc[start:pos]
                df_win[["open","high","low","close"]] = df_win[["open","high","low","close"]].astype("float64")
                
                end_str = end.strftime("%Y-%m-%d")
                blob_name = f"{symbol}/window_{w}_{ema_suffix}/{end_str}.png"

                # Skip if already exists
                if blob_name in existing_blobs:
                    skipped += 1
                    continue

                # EMA addplot 생성
                addplots = [mpf.make_addplot(df_win[f"EMA_{ema}"], color='#ef5714', width=2.0) for ema in emas]

                plot_kwargs = dict(
                    type=chart_type,
                    style="charles",
                    volume=False,
                    figsize=figsize,
                    returnfig=True,
                )
                if addplots:
                    plot_kwargs["addplot"] = addplots

                # Generate chart
                fig, _ = mpf.plot(df_win, **plot_kwargs)

                # Save to BytesIO
                img_buffer = BytesIO()
                fig.savefig(img_buffer, dpi=dpi, bbox_inches="tight", pad_inches=0, pil_kwargs={"compress_level": 1})
                plt.close(fig)
                img_buffer.seek(0)

                # Resize
                img = Image.open(img_buffer)
                img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img_bytes = BytesIO()
                img_resized.save(img_bytes, format='PNG', compress_level=1)
                img_bytes.seek(0)

                # Upload to GCS
                bucket.blob(blob_name).upload_from_file(img_bytes, content_type="image/png")

                saved += 1
                if saved % 50 == 0:
                    dt = time.time() - t0
                    print(f"  {saved} saved, {skipped} skipped, elapsed={dt:.1f}s last={end_str}")

            dt = time.time() - t0
            print(f"  Completed: {saved} saved, {skipped} skipped, elapsed={dt:.1f}s")


def run_daily_chart_pipeline(ds, windows, emas_list, chart_type, image_size):
    """
    대상 심볼 리스트에 대해 데이터를 로드하고, 차트를 생성하여 GCS에 저장하는 전체 프로세스를 실행
    """
    yesterday = datetime.strptime(ds, "%Y-%m-%d").date()
    df = load_all_data(yesterday)

    for sym, group in df.groupby("symbol"):
        generate_and_upload_chart(
            df=group,
            windows=windows,
            emas_list=emas_list,
            end_start=group.index.max(),
            end_stop=group.index.min() + timedelta(days=90),
            bucket_name=bucket_name,
            symbol=sym,
            chart_type=chart_type,
            image_size=image_size
        )


default_args = {
    "owner": "jeong",
    "depends_on_past": True,
    "start_date": datetime(2025, 11, 1),
    "end_date": datetime(2026, 2, 14),
}

with DAG(
    dag_id="price-ETL",
    default_args=default_args,
    schedule_interval='45 13 * * *',
    tags=['my_dags'],
    catchup=True,
) as dag:
    execution_date = "{{ ds }}" # 지정된 template 사용
    
    extract_task = PythonOperator(
        task_id="load_ohlcv",
        python_callable=load_ohlcv,
        op_kwargs = {
            "date": execution_date
        }
    )
    
    check_task = PythonOperator(
        task_id="check_stock_data",
        python_callable=check_stock_data,
        op_kwargs = {
            "date": execution_date
        }
    )
    
    trasnform_task = PythonOperator(
        task_id="feature_engineering_and_store_datasets",
        python_callable=split_and_store_datasets,
        op_kwargs = {
            "date": execution_date
        }
    )
    
    extract_task >> check_task >> trasnform_task


# 지금은 제외

"""
chart_task = PythonOperator(
        task_id="make_daily_chart",
        python_callable=run_daily_chart_pipeline,
        op_kwargs = {
            "date": execution_date,
            "windows": [5, 20, 60],
            "emas_list": [[5, 20]],
            "chart_type": "candle",
            "image_size": 448,
        }
    )

extract_task >> check_task >> trasnform_task >> chart_task
"""
