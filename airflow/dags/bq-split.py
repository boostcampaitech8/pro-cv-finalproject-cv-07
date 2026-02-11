import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from dotenv import load_dotenv


load_dotenv("/data/ephemeral/home/airflow/.env")

PROJECT_ID = os.getenv("BIGQUERY_PROJECT")
DATASET_ID = os.getenv("BIGQUERY_DATASET")

PYTHON_BIN = os.getenv("AIRFLOW_PYTHON_BIN", "python")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/data/ephemeral/home/pro-cv-finalproject-cv-07")

SCRIPTS_DIR = f"{PROJECT_ROOT}/python/scripts"
OUTPUT_DIR = f"{PROJECT_ROOT}/python/src/datasets/bq_splits"

COMMODITIES = os.getenv("COMMODITIES", "corn,wheat,soybean,gold,silver,copper").split(",")


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
    dag_id="bq-split",
    default_args=default_args,
    schedule_interval="45 13 * * *",
    catchup=True,
    tags=["my_dags"],
) as dag:
    execution_date = "{{ ds }}"

    wait_candle_sync = ExternalTaskSensor(
        task_id="wait_candle_sync",
        external_dag_id="candle-sync",
        external_task_id="sync_candle_images",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        execution_delta=timedelta(0),
        mode="reschedule",
        poke_interval=60,
        timeout=60 * 60,
    )

    build_split = BashOperator(
        task_id="build_bq_split",
        bash_command=(
            f"cd {PROJECT_ROOT}/python && "
            f"{PYTHON_BIN} {SCRIPTS_DIR}/build_bq_split.py "
            f"--project_id {PROJECT_ID} "
            f"--dataset_id {DATASET_ID} "
            f"--commodities " + " ".join(COMMODITIES) + " "
            f"--output_dir {OUTPUT_DIR}"
        ),
    )

    wait_candle_sync >> build_split
