"""
Report Generation Pipeline
- BigQuery에서 예측 가격 조회
- GCS에서 예측 차트 이미지 다운로드
- GPT API로 마크다운 리포트 생성
- MySQL에 리포트 저장
"""

import os
import sys
import base64
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pymysql
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from openai import OpenAI

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
REPORT_DIR = Path(__file__).resolve().parent          # plugins/report/
AIRFLOW_ROOT = REPORT_DIR.parent.parent               # airflow/
REPO_ROOT = AIRFLOW_ROOT.parent
sys.path.insert(0, str(REPORT_DIR))

from prompts import REPORT_SYSTEM_PROMPT, build_report_prompt

# ---------------------------------------------------------------------------
# BigQuery – 예측 가격 조회
# ---------------------------------------------------------------------------

def _get_price_client() -> bigquery.Client:
    key_path = os.environ["PRICE_GOOGLE_APPLICATION_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_file(key_path)
    project = os.environ["PRICE_BIGQUERY_PROJECT"]
    return bigquery.Client(credentials=credentials, project=project)


def fetch_predicted_prices(base_date: str) -> pd.DataFrame:
    """BigQuery predict_price 테이블에서 base_date 기준 예측 가격을 가져온다.

    Returns:
        DataFrame with columns: base_date, type, t, t+1 ... t+19
    """
    project = os.environ["PRICE_BIGQUERY_PROJECT"]
    dataset = os.environ["PRICE_BIGQUERY_DATASET"]
    table = f"{project}.{dataset}.predict_price"

    query = f"""
        SELECT *
        FROM `{table}`
        WHERE base_date = @base_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("base_date", "STRING", base_date),
        ]
    )

    client = _get_price_client()
    df = client.query(query, job_config=job_config).to_dataframe()
    print(f"[Report] Fetched {len(df)} rows from predict_price (base_date={base_date})")
    return df


def _prices_to_text(df: pd.DataFrame) -> str:
    """DataFrame을 프롬프트 삽입용 텍스트로 변환."""
    if df.empty:
        return "(예측 가격 데이터 없음)"
    return df.to_string(index=False)


# ---------------------------------------------------------------------------
# GCS – 예측 이미지 다운로드
# ---------------------------------------------------------------------------

def fetch_prediction_images(base_date: str) -> list[dict]:
    """GCS에서 DeepAR/TFT 예측 차트 6장을 다운로드하여 base64로 반환.

    Returns:
        [{"label": "deepAR_w5", "base64": "..."}, ...]
    """
    bucket_name = os.environ["GCS_AI_BUCKET"]
    key_path = os.environ["PRICE_GOOGLE_APPLICATION_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_file(key_path)
    gcs_client = storage.Client(credentials=credentials)
    bucket = gcs_client.bucket(bucket_name)

    models = ["deepAR", "TFT"]
    windows = ["w5", "w20", "w60"]
    images: list[dict] = []

    for model in models:
        for window in windows:
            blob_path = f"{base_date}/{model}/{window}.png"
            blob = bucket.blob(blob_path)

            if not blob.exists():
                print(f"[Report] WARNING: image not found – gs://{bucket_name}/{blob_path}")
                continue

            data = blob.download_as_bytes()
            images.append({
                "label": f"{model}_{window}",
                "base64": base64.b64encode(data).decode("utf-8"),
            })
            print(f"[Report] Downloaded gs://{bucket_name}/{blob_path}")

    print(f"[Report] Total images fetched: {len(images)}")
    return images


# ---------------------------------------------------------------------------
# GPT API – 리포트 생성
# ---------------------------------------------------------------------------

def generate_report(price_text: str, images: list[dict]) -> str:
    """OpenAI API로 마크다운 리포트를 생성한다."""
    client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

    # 유저 메시지 구성: 텍스트 + 이미지
    user_content: list[dict] = [
        {"type": "text", "text": build_report_prompt(price_text)},
    ]
    for img in images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['base64']}",
                "detail": "high",
            },
        })

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4096,
        temperature=0.3,
    )

    report = response.choices[0].message.content
    print(f"[Report] Generated report ({len(report)} chars)")
    return report


# ---------------------------------------------------------------------------
# MySQL – 리포트 저장
# ---------------------------------------------------------------------------

def save_report_to_mysql(publish_date: str, base_date: str, content: str):
    """MySQL report 테이블에 리포트를 저장한다."""
    conn = pymysql.connect(
        host=os.environ["REPORT_DB_HOST"],
        port=int(os.environ.get("REPORT_DB_PORT", 3306)),
        user=os.environ["REPORT_DB_USER"],
        password=os.environ["REPORT_DB_PASS"],
        database=os.environ["REPORT_DB_NAME"],
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cursor:
            sql = (
                "INSERT INTO report (publish_date, base_date, created_at, content) "
                "VALUES (%s, %s, %s, %s)"
            )
            cursor.execute(sql, (publish_date, base_date, datetime.now(timezone.utc), content))
        conn.commit()
        print(f"[Report] Saved report to MySQL (publish_date={publish_date})")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 오케스트레이터
# ---------------------------------------------------------------------------

def run_report_pipeline(publish_date: str, base_date: str):
    """리포트 생성 파이프라인 실행."""
    print(f"[Report] Pipeline start – publish_date={publish_date}, base_date={base_date}")

    # 1) 예측 가격 조회
    price_df = fetch_predicted_prices(base_date)
    price_text = _prices_to_text(price_df)

    # 2) 예측 이미지 다운로드
    images = fetch_prediction_images(base_date)

    # 3) GPT 리포트 생성
    report_md = generate_report(price_text, images)

    # 4) MySQL 저장
    save_report_to_mysql(publish_date, base_date, report_md)

    print("[Report] Pipeline complete")
