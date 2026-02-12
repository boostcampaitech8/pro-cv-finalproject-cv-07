import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from dotenv import load_dotenv


load_dotenv("/data/ephemeral/home/airflow/.env")

PROJECT_ID = os.getenv("BIGQUERY_PROJECT")
DATASET_ID = os.getenv("BIGQUERY_DATASET")
BUCKET_NAME = os.getenv("CANDLE_BUCKET", "boostcamp-final-proj")

PYTHON_BIN = os.getenv("AIRFLOW_PYTHON_BIN", "python")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/data/ephemeral/home/pro-cv-finalproject-cv-07")
PREDICT_ROOT = f"{PROJECT_ROOT}/predict"
SCRIPTS_DIR = f"{PREDICT_ROOT}/CNN/scripts"

SYMBOLS = os.getenv("CANDLE_SYMBOLS", "ZC=F,ZW=F,ZS=F,GC=F,SI=F,HG=F").split(",")
WINDOWS = [int(x) for x in os.getenv("CANDLE_WINDOWS", "5,20,60").split(",")]
CHART_TYPE = os.getenv("CANDLE_CHART_TYPE", "candle")
IMAGE_SIZE = int(os.getenv("CANDLE_IMAGE_SIZE", "224"))


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


START_DATE = _parse_date(os.getenv("AIRFLOW_START_DATE")) or datetime(2026, 2, 8)
END_DATE = _parse_date(os.getenv("AIRFLOW_END_DATE")) or datetime(2026, 2, 14)


default_args = {
    "owner": "jeong",
    "depends_on_past": True,
    "start_date": START_DATE,
    "end_date": END_DATE,
}


with DAG(
    dag_id="price-OHLC-image",
    default_args=default_args,
    schedule_interval="45 13 * * *",
    catchup=True,
    tags=["my_dags"],
) as dag:
    execution_date = "{{ ds }}"

    wait_price_etl = ExternalTaskSensor(
        task_id="wait_price_etl",
        external_dag_id="price-ETL",
        external_task_id="feature_engineering_and_store_datasets",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        execution_delta=timedelta(0),
        mode="reschedule",
        poke_interval=60,
        timeout=60 * 60,
    )

    sync_task = BashOperator(
        task_id="sync_candle_images",
        bash_command=(
            f"cd {PREDICT_ROOT} && "
            f"{PYTHON_BIN} {SCRIPTS_DIR}/candle_sync.py "
            f"--ds {execution_date} "
            f"--project_id {PROJECT_ID} "
            f"--dataset_id {DATASET_ID} "
            f"--bucket {BUCKET_NAME} "
            f"--symbols " + " ".join(SYMBOLS) + " "
            f"--windows " + " ".join(map(str, WINDOWS)) + " "
            f"--image_size {IMAGE_SIZE} "
            f"--chart_type {CHART_TYPE} "
            f"--dotenv_path /data/ephemeral/home/airflow/.env"
        ),
    )

    wait_price_etl >> sync_task
