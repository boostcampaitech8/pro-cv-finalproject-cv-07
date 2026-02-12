"""
Article ETL DAG
- 매일 13:45 UTC에 뉴스 수집·가공·저장 파이프라인 실행
- 크롤링 → T/F 판단 → KG 추출 → ID 생성 → Embedding → Features → DB 저장
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# plugins/article, connections 경로 추가
_AIRFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AIRFLOW_ROOT / 'plugins' / 'article'))
sys.path.insert(0, str(_AIRFLOW_ROOT / 'connections'))


def _run_pipeline(**context):
    """run_pipeline을 DAG 실행 컨텍스트에 맞게 호출"""
    from main_article import run_pipeline

    # data_interval_end = DAG이 실제 트리거되는 시점의 날짜
    end_date_str = context['data_interval_end'].strftime('%Y-%m-%d')

    run_pipeline(
        days_back=1,
        start_hour=13,
        start_minute=45,
        end_date_str=end_date_str,
    )


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='article_ETL',
    default_args=default_args,
    description='뉴스 수집·가공·저장 파이프라인 (daily)',
    schedule='45 13 * * *',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['article', 'ETL'],
) as dag:

    run_article_pipeline = PythonOperator(
        task_id='run_article_pipeline',
        python_callable=_run_pipeline,
    )
