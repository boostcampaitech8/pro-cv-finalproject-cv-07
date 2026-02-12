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
from torch.utils.data import DataLoader

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
from data.dataset_cnn import CNNDataset, cnn_collate_fn, HORIZONS
from shared.data.bigquery_loader import load_news_features_bq
from engine.trainer_cnn import train_cnn
from models.CNN import CNN
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
    checkpoint_root = _format_root(cfg.checkpoint_root, cfg.target_commodity, date_tag, cfg.output_tag)

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
            print(f"▶ CNN TRAIN window={window_size} fold={fold}")
            print("=" * 60)

            exp_name = _build_exp_name(cfg, window_size, fold)

            ds_kwargs = dict(
                commodity=cfg.target_commodity,
                fold=fold,
                window_size=window_size,
                image_mode=cfg.image_mode,
                use_aux=cfg.use_aux,
                aux_type=cfg.aux_type,
                data_dir=str(data_dir),
                split_file=str(split_path),
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

            ckpt_dir = checkpoint_root / f"w{window_size}"
            checkpoint_path = ckpt_dir / "best_model.pt"
            metrics_path = ckpt_dir / "val_metrics.json"
            log_path = ckpt_dir / "train_log.jsonl" if cfg.save_train_log else None

            if cfg.loss == "smooth_l1":
                loss_fn = torch.nn.SmoothL1Loss(reduction="none")
            else:
                loss_fn = torch.nn.MSELoss(reduction="none")

            train_ds = CNNDataset(split="train", **ds_kwargs)
            val_ds = CNNDataset(split="val", **ds_kwargs)

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                collate_fn=cnn_collate_fn,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=cnn_collate_fn,
            )

            in_chans = 3 if cfg.image_mode == "stack" else 2 if cfg.image_mode in {"candle_gaf", "candle_rp"} else 1
            aux_dim = train_ds.news_dim if cfg.use_aux else 0

            num_outputs = len(HORIZONS)
            horizon_weights = cfg.horizon_weights
            if horizon_weights is not None:
                if len(horizon_weights) == 1:
                    horizon_weights = horizon_weights * num_outputs
                elif len(horizon_weights) != num_outputs:
                    raise ValueError(
                        f"horizon_weights length {len(horizon_weights)} does not match horizons {num_outputs}"
                    )

            model = CNN(
                backbone=cfg.backbone,
                in_chans=in_chans,
                aux_dim=aux_dim,
                fusion=cfg.fusion,
                dropout=0.1,
                num_outputs=num_outputs,
                pretrained=True,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

            train_cnn(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epochs=cfg.epochs,
                early_stop_patience=cfg.early_stop_patience,
                horizon_weights=horizon_weights,
                severity_loss_weight=cfg.severity_loss_weight,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
                metrics_path=metrics_path,
                best_metric=cfg.early_stop_metric,
                min_epochs=cfg.min_epochs,
                freeze_backbone_epochs=cfg.freeze_backbone_epochs,
                meta={
                    "commodity": cfg.target_commodity,
                    "fold": fold,
                    "window_size": window_size,
                },
            )

            print(f"✓ Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main(tyro.cli(CNNBatchConfig))
