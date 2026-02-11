import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from dotenv import load_dotenv


def _load_env() -> None:
    candidates = [
        os.getenv("AIRFLOW_ENV"),
        "/data/ephemeral/home/pro-cv-finalproject-cv-07/airflow/.env",
        "/data/ephemeral/home/airflow/.env",
    ]
    for path in candidates:
        if path and Path(path).exists():
            load_dotenv(path)
            return


_load_env()

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/data/ephemeral/home/pro-cv-finalproject-cv-07")
PYTHON_BIN = os.getenv("AIRFLOW_PYTHON_BIN", "/data/ephemeral/home/airflow/bin/python")

COMMODITIES = os.getenv("COMMODITIES", "corn,wheat,soybean,gold,silver,copper").split(",")
WINDOW_SIZES = os.getenv("WINDOW_SIZES", "5 20 60").split()

BQ_PROJECT = "esoteric-buffer-485608-g5"
BQ_DATASET = "final_proj"
BQ_TABLE = os.getenv("BQ_PREDICT_TABLE", "predict_price")

GCS_BUCKET = "boostcamp-final-proj"
GCS_PREFIX = "predict-{symbol}-w{window}"
GCS_NAME_PATTERN = "predict-{symbol}-{suffix}.png"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


START_DATE = _parse_date(os.getenv("AIRFLOW_START_DATE")) or datetime(2026, 2, 8)
END_DATE = _parse_date(os.getenv("AIRFLOW_END_DATE"))

default_args = {
    "owner": "jeong",
    "depends_on_past": True,
    "start_date": START_DATE,
    "end_date": END_DATE,
}


def _join(items: list[str]) -> str:
    return " ".join([item.strip() for item in items if item.strip()])


COMMODITY_ARGS = _join(COMMODITIES)
WINDOW_ARGS = _join(WINDOW_SIZES)


with DAG(
    dag_id="publish-outputs",
    default_args=default_args,
    schedule_interval="45 13 * * *",
    catchup=True,
    tags=["my_dags"],
) as dag:
    wait_model_train = ExternalTaskSensor(
        task_id="wait_model_train",
        external_dag_id="model-train",
        external_task_id="all_done",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        execution_delta=timedelta(0),
        mode="reschedule",
        poke_interval=60,
        timeout=60 * 60,
    )

    publish_outputs = BashOperator(
        task_id="publish_outputs",
        bash_command=(
            f"cd {PROJECT_ROOT}/python && "
            f"TRUNC=1; TRUNC_SQL=1; "
            f"for c in {COMMODITY_ARGS}; do "
            f"TFT_DIR=$(ls -td src/outputs/predictions/${{c}}_*_tft_eval 2>/dev/null | head -1); "
            f"DEE_DIR=$(ls -td src/outputs/predictions/${{c}}_*_deepar_eval 2>/dev/null | head -1); "
            f"CNN_DIR=$(ls -td src/outputs/predictions/${{c}}_*_cnn_eval 2>/dev/null | head -1); "
            f"if [ -z \"$TFT_DIR\" ] || [ -z \"$DEE_DIR\" ] || [ -z \"$CNN_DIR\" ]; then "
            f"echo \"Missing predictions for ${{c}}\"; continue; fi; "
            f"for w in {WINDOW_ARGS}; do "
            f"EXTRA=''; if [ \"$TRUNC\" = 1 ]; then EXTRA='--truncate_table'; TRUNC=0; fi; "
            f"EXTRA_SQL=''; if [ \"$TRUNC_SQL\" = 1 ]; then EXTRA_SQL='--truncate_sql'; TRUNC_SQL=0; fi; "
            f"{PYTHON_BIN} scripts/publish_outputs.py "
            f"--symbol ${{c}} "
            f"--window ${{w}} "
            f"--tft_predictions $TFT_DIR/tft_predictions.csv "
            f"--deepar_predictions $DEE_DIR/deepar_predictions.csv "
            f"--cnn_predictions $CNN_DIR/cnn_predictions.csv "
            f"--data_dir src/datasets/local_bq_like/${{c}} "
            f"--output_dir src/outputs/predictions/${{c}}_bundle/w${{w}} "
            f"--gcs_bucket {GCS_BUCKET} "
            f"--gcs_prefix \"{GCS_PREFIX}\" "
            f"--gcs_name_pattern \"{GCS_NAME_PATTERN}\" "
            f"--project_id {BQ_PROJECT} "
            f"--dataset_id {BQ_DATASET} "
            f"--table_id {BQ_TABLE} "
            f"$EXTRA $EXTRA_SQL "
            f"--upload_bq --upload_gcs --upload_sql; "
            f"done; "
            f"done"
        ),
        env=os.environ.copy(),
    )

    wait_model_train >> publish_outputs
