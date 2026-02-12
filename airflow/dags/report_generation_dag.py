"""
Report Generation DAG
- 매일 15:00 UTC에 리포트 생성 파이프라인 실행
- BigQuery 예측 가격 + GCS 차트 이미지 → GPT 리포트 → MySQL 저장
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# plugins/report 경로 추가
_AIRFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AIRFLOW_ROOT / "plugins" / "report"))


def _run_report(**context):
    """run_report_pipeline을 DAG 실행 컨텍스트에 맞게 호출."""
    from report_generation import run_report_pipeline

    # publish_date = DAG이 실제 트리거되는 시점의 날짜
    publish_date = context["data_interval_end"].strftime("%Y-%m-%d")
    # base_date = publish_date - 1일
    base_date = (context["data_interval_end"] - timedelta(days=1)).strftime("%Y-%m-%d")

    run_report_pipeline(publish_date=publish_date, base_date=base_date)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="report_generation",
    default_args=default_args,
    description="예측 리포트 생성 파이프라인 (daily)",
    schedule="0 14 * * *", # 임의로 지정
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["report", "generation"],
) as dag:

    run_report = PythonOperator(
        task_id="run_report_pipeline",
        python_callable=_run_report,
    )
