import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
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
PYTHON_BIN = os.getenv("AIRFLOW_PYTHON_BIN", "/data/ephemeral/home/TFT/bin/python")

SPLIT_DIR = f"{PROJECT_ROOT}/python/src/datasets/bq_splits"
DATA_ROOT = f"{PROJECT_ROOT}/python/src/datasets/local_bq_like"

COMMODITIES = os.getenv("COMMODITIES", "corn,wheat,soybean,gold,silver,copper").split(",")
WINDOW_SIZES = os.getenv("WINDOW_SIZES", "5 20 60")


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


def _comma_join(items: list[str]) -> str:
    return " ".join([item.strip() for item in items if item.strip()])


COMMODITY_ARGS = _comma_join(COMMODITIES)


with DAG(
    dag_id="model-train",
    default_args=default_args,
    schedule_interval="45 13 * * *",
    catchup=True,
    tags=["my_dags"],
) as dag:
    wait_bq_split = ExternalTaskSensor(
        task_id="wait_bq_split",
        external_dag_id="bq-split",
        external_task_id="build_bq_split",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        execution_delta=timedelta(0),
        mode="reschedule",
        poke_interval=60,
        timeout=60 * 60,
    )

    tft_train = BashOperator(
        task_id="train_tft",
        bash_command=(
            f"cd {PROJECT_ROOT}/python && "
            f"for c in {COMMODITY_ARGS}; do "
            f"{PYTHON_BIN} scripts/run_tft_batch.py "
            f"--data_source bigquery "
            f"--split_file {SPLIT_DIR}/${{c}}_split.json "
            f"--target_commodity ${{c}} "
            f"--seq_lengths {WINDOW_SIZES} "
            f"--fold 0 "
            f"--do_infer "
            f"--checkpoint_root \"src/outputs/checkpoints/{{commodity}}_{{date}}_tft\" "
            f"--prediction_root \"src/outputs/predictions/{{commodity}}_{{date}}_tft\"; "
            f"done"
        ),
    )

    deepar_train = BashOperator(
        task_id="train_deepar",
        bash_command=(
            f"cd {PROJECT_ROOT}/python && "
            f"for c in {COMMODITY_ARGS}; do "
            f"{PYTHON_BIN} scripts/run_deepar_batch.py "
            f"--data_source bigquery "
            f"--split_file {SPLIT_DIR}/${{c}}_split.json "
            f"--target_commodity ${{c}} "
            f"--seq_lengths {WINDOW_SIZES} "
            f"--fold 0 "
            f"--no-do_test "
            f"--checkpoint_root \"src/outputs/checkpoints/{{commodity}}_{{date}}_deepar\" "
            f"--prediction_root \"src/outputs/predictions/{{commodity}}_{{date}}_deepar\"; "
            f"done"
        ),
    )

    cnn_train = BashOperator(
        task_id="train_cnn",
        bash_command=(
            f"cd {PROJECT_ROOT}/python && "
            f"for c in {COMMODITY_ARGS}; do "
            f"{PYTHON_BIN} scripts/run_cnn_batch.py "
            f"--data_dir {DATA_ROOT}/${{c}} "
            f"--split_file {SPLIT_DIR}/${{c}}_split.json "
            f"--target_commodity ${{c}} "
            f"--window_sizes {WINDOW_SIZES} "
            f"--folds 0 "
            f"--output_tag \"\"; "
            f"done"
        ),
    )

    all_done = EmptyOperator(
        task_id="all_done",
        trigger_rule="all_success",
    )

    wait_bq_split >> [tft_train, deepar_train, cnn_train] >> all_done
