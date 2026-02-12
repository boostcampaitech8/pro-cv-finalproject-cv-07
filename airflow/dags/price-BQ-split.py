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
PREDICT_ROOT = f"{PROJECT_ROOT}/predict"

SCRIPTS_DIR = f"{PREDICT_ROOT}/shared/scripts/preprocessing"
OUTPUT_DIR = f"{PREDICT_ROOT}/shared/src/datasets/bq_splits"

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
    dag_id="price-BQ-split",
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

    build_split = BashOperator(
        task_id="build_bq_split",
        bash_command=(
            f"cd {PREDICT_ROOT} && "
            "for c in " + " ".join(COMMODITIES) + "; do "
            f"{PYTHON_BIN} {SCRIPTS_DIR}/build_split.py "
            f"--data_source bigquery "
            f"--bq_project_id {PROJECT_ID} "
            f"--bq_dataset_id {DATASET_ID} "
            f"--bq_train_table train_price "
            f"--target_commodity $c "
            f"--val_months 3 "
            f"--output_dir {OUTPUT_DIR}; "
            "done"
        ),
    )

    wait_price_etl >> build_split
