from __future__ import annotations

import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
import tyro
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
PREDICT_ROOT = MODEL_ROOT.parent
MODEL_SRC = MODEL_ROOT / "src"
for _path in (MODEL_SRC, PREDICT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_ROOT = PREDICT_ROOT.parent

# Suppress noisy runtime warnings in batch runs
warnings.filterwarnings("ignore")

from configs.cnn_config import CNNBatchConfig
from shared.data.bigquery_loader import load_news_features_bq
from engine.inference_cnn import run_inference_cnn
from shared.utils.set_seed import set_seed

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _strip_quotes(value: str) -> str:
    if not value:
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = _strip_quotes(val.strip())
    return env


def _resolve_price_bq_settings(cfg: CNNBatchConfig) -> dict[str, str | None]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> str | None:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("PRICE_BIGQUERY_PROJECT") or cfg.bq_project_id,
        "dataset_id": _get("PRICE_BIGQUERY_DATASET") or cfg.bq_dataset_id,
        "credentials_path": _get("PRICE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def _resolve_article_bq_settings(cfg: CNNBatchConfig) -> dict[str, str | None]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> str | None:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("ARTICLE_BIGQUERY_PROJECT") or cfg.bq_news_project_id,
        "dataset_id": _get("ARTICLE_BIGQUERY_DATASET") or cfg.bq_news_dataset_id,
        "credentials_path": _get("ARTICLE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "daily_table": _get("TABLE_DAILY_SUMMARY") or cfg.bq_news_table,
    }


def _resolve_gcs_bucket(cfg: CNNBatchConfig) -> str | None:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> str | None:
        return os.getenv(key) or env_file.get(key)

    return (
        _get("GCS_AI_BUCKET")
        or _get("GCS_SERVER_BUCKET")
        or _get("CANDLE_BUCKET")
        or cfg.gcs_bucket
    )


def _infer_date_tag_from_split(split_file: str) -> str | None:
    if not split_file:
        return None
    path = Path(split_file)
    if not path.exists():
        return None
    try:
        data = json.load(path.open("r"))
    except Exception:
        return None

    preferred_keys = {
        "inference_end",
        "inference_end_date",
        "inference_end_dt",
        "test_end",
        "test_end_date",
        "as_of",
        "base_date",
    }
    candidates: list[str] = []

    def visit(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and _DATE_RE.match(v):
                    if k in preferred_keys:
                        candidates.append(v)
                    else:
                        candidates.append(v)
                else:
                    visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    visit(data)
    if not candidates:
        return None
    return max(candidates)


def _fetch_latest_inference_date(
    *,
    data_source: str,
    data_root: Path,
    project_id: str | None,
    dataset_id: str | None,
    table: str | None,
    credentials_path: str | None = None,
) -> str | None:
    if data_source == "bigquery":
        if not project_id or not dataset_id or not table:
            return None
        try:
            from google.cloud import bigquery
        except Exception:
            return None

        creds = None
        if credentials_path:
            try:
                cred_path = Path(credentials_path)
                if cred_path.is_file():
                    from google.oauth2 import service_account

                    creds = service_account.Credentials.from_service_account_file(str(cred_path))
            except Exception:
                creds = None

        client = bigquery.Client(project=project_id, credentials=creds)
        query = f"SELECT MAX(trade_date) AS max_date FROM `{project_id}.{dataset_id}.{table}`"
        try:
            rows = list(client.query(query).result())
        except Exception:
            return None
        if not rows:
            return None
        max_date = rows[0].get("max_date")
        if max_date is None:
            return None
        return str(max_date)[:10]

    infer_path = data_root / "inference_price.csv"
    if not infer_path.exists():
        return None
    try:
        df = pd.read_csv(infer_path)
    except Exception:
        return None
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return None
    max_dt = pd.to_datetime(df[date_col], errors="coerce").max()
    if pd.isna(max_dt):
        return None
    return max_dt.strftime("%Y-%m-%d")


def _infer_date_tag(split_path: Path) -> str:
    if not split_path.exists():
        return datetime.now().strftime("%Y-%m-%d")
    try:
        payload = json.loads(split_path.read_text())
        meta = payload.get("meta", {})
        if isinstance(meta, dict):
            inference_window = meta.get("inference_window", {})
            for key in ("end", "end_date"):
                if key in inference_window:
                    return str(inference_window[key])[:10]
            for key in ("test_end_date", "end_date"):
                if key in meta:
                    return str(meta[key])[:10]
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")


def _format_root(template: str, commodity: str, date_tag: str, output_tag: str) -> Path:
    tag = f"_{output_tag}" if output_tag else ""
    return Path(template.format(commodity=commodity, date=date_tag, tag=tag))


def _build_exp_name(cfg: CNNBatchConfig, window_size: int, fold: int) -> str:
    if cfg.exp_name:
        return cfg.exp_name
    aux_flag = "aux" if cfg.use_aux else "noaux"
    return (
        f"{cfg.backbone}_{cfg.image_mode}_w{window_size}_"
        f"fold{fold}_{cfg.fusion}_{aux_flag}"
    )


def main(cfg: CNNBatchConfig) -> None:
    set_seed(cfg.seed)

    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = PREDICT_ROOT / path
        return str(path)

    cfg.data_dir = _resolve_path(cfg.data_dir)
    cfg.checkpoint_root = _resolve_path(cfg.checkpoint_root)
    cfg.prediction_root = _resolve_path(cfg.prediction_root)

    # Resolve .env overrides (PRICE_/ARTICLE_/GCS)
    price_bq = _resolve_price_bq_settings(cfg)
    article_bq = _resolve_article_bq_settings(cfg)
    gcs_bucket = _resolve_gcs_bucket(cfg)

    if price_bq.get("credentials_path") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(price_bq["credentials_path"])

    if price_bq.get("project_id"):
        cfg.bq_project_id = str(price_bq["project_id"])
    if price_bq.get("dataset_id"):
        cfg.bq_dataset_id = str(price_bq["dataset_id"])
    if article_bq.get("project_id"):
        cfg.bq_news_project_id = str(article_bq["project_id"])
    if article_bq.get("dataset_id"):
        cfg.bq_news_dataset_id = str(article_bq["dataset_id"])
    if article_bq.get("daily_table"):
        cfg.bq_news_table = str(article_bq["daily_table"])
    if gcs_bucket:
        cfg.gcs_bucket = str(gcs_bucket)

    data_dir = Path(cfg.data_dir).resolve()
    split_file = cfg.split_file
    if "{commodity}" in split_file:
        split_file = split_file.format(commodity=cfg.target_commodity)
    split_path = Path(split_file)
    if not split_path.is_absolute():
        candidate = PREDICT_ROOT / split_path
        if candidate.exists():
            split_path = candidate
        else:
            split_path = (data_dir / split_path).resolve()
    else:
        split_path = split_path.resolve()

    latest_infer_date = _fetch_latest_inference_date(
        data_source=cfg.data_source,
        data_root=data_dir,
        project_id=cfg.bq_project_id,
        dataset_id=cfg.bq_dataset_id,
        table=cfg.bq_inference_table,
        credentials_path=price_bq.get("credentials_path"),
    )
    date_tag = (
        cfg.date_tag
        or latest_infer_date
        or _infer_date_tag_from_split(str(split_path))
        or _infer_date_tag(split_path)
    )
    prediction_root = _format_root(cfg.prediction_root, cfg.target_commodity, date_tag, cfg.output_tag)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    news_df = None
    if cfg.use_aux and cfg.aux_type == "news" and cfg.news_source == "bigquery":
        news_df = load_news_features_bq(
            project_id=cfg.bq_news_project_id,
            dataset_id=cfg.bq_news_dataset_id,
            table=cfg.bq_news_table,
            commodity=cfg.target_commodity,
            credentials_path=article_bq.get("credentials_path"),
        )

    for fold in cfg.folds:
        for window_size in cfg.seq_lengths:
            print("\n" + "=" * 60)
            print(f"▶ CNN INFER window={window_size} fold={fold}")
            print("=" * 60)

            exp_name = _build_exp_name(cfg, window_size, fold)

            ckpt_dir = Path(
                cfg.checkpoint_root.format(
                    commodity=cfg.target_commodity,
                    date=date_tag,
                    tag=f"_{cfg.output_tag}" if cfg.output_tag else "",
                )
            ) / f"w{window_size}"
            checkpoint_path = ckpt_dir / "best_model.pt"
            if not checkpoint_path.exists():
                print(f"⚠️  Checkpoint not found (skip): {checkpoint_path}")
                continue

            run_inference_cnn(
                commodity=cfg.target_commodity,
                fold=fold,
                split=cfg.inference_split,
                window_size=window_size,
                image_mode=cfg.image_mode,
                backbone=cfg.backbone,
                fusion=cfg.fusion,
                exp_name=exp_name,
                use_aux=cfg.use_aux,
                aux_type=cfg.aux_type,
                checkpoint_path=str(checkpoint_path),
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                device=device,
                save_gradcam=False,
                data_dir=str(data_dir),
                split_file=str(split_path),
                prediction_root=str(prediction_root),
                date_tag=date_tag,
                latest_only=not cfg.all_dates,
                write_json=cfg.write_json,
                write_csv=True,
                news_data=news_df,
                data_source=cfg.data_source,
                bq_project_id=cfg.bq_project_id,
                bq_dataset_id=cfg.bq_dataset_id,
                bq_train_table=cfg.bq_train_table,
                bq_inference_table=cfg.bq_inference_table,
                bq_test_table=cfg.bq_test_table,
                image_source=cfg.image_source,
                gcs_bucket=cfg.gcs_bucket,
                gcs_prefix_template=cfg.gcs_prefix_template,
            )

            csv_path = prediction_root / "cnn_predictions.csv"
            if csv_path.exists():
                print(f"✓ Saved predictions: {csv_path}")
            else:
                print(f"⚠️  Predictions CSV not found: {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(CNNBatchConfig))
