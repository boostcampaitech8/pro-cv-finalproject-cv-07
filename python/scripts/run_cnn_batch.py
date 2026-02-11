from __future__ import annotations

import json
import warnings
import re
from datetime import datetime
from pathlib import Path

# Suppress noisy runtime warnings in batch runs
warnings.filterwarnings("ignore")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _infer_date_tag_from_split(split_file: str) -> str | None:
    """
    Try to infer date_tag from split json (prefer inference/test end dates).
    """
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
import sys
from datetime import datetime
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silence noisy gluonts JSON warnings during training logs.
warnings.filterwarnings(
    "ignore",
    message=r"Using `json`-module for json-handling.*",
    category=UserWarning,
)

from src.configs.cnn_config import CNNBatchConfig
from src.data.dataset_cnn import CNNDataset, cnn_collate_fn, HORIZONS
from src.data.bigquery_loader import load_news_features_bq
from src.engine.inference_cnn import run_inference_cnn
from src.engine.trainer_cnn import train_cnn, evaluate_cnn
from src.metrics.metrics_cnn import summarize_metrics
from src.models.CNN import CNN
from src.utils.set_seed import set_seed


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

    data_dir = Path(cfg.data_dir).resolve()
    split_path = Path(cfg.split_file)
    if not split_path.is_absolute():
        candidate = PROJECT_ROOT / split_path
        if candidate.exists():
            split_path = candidate
        else:
            split_path = (data_dir / split_path).resolve()
    else:
        split_path = split_path.resolve()

    date_tag = cfg.date_tag or _infer_date_tag(split_path)
    checkpoint_root = _format_root(cfg.checkpoint_root, cfg.target_commodity, date_tag, cfg.output_tag)
    prediction_root = _format_root(cfg.prediction_root, cfg.target_commodity, date_tag, cfg.output_tag)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    news_df = None
    if cfg.use_aux and cfg.aux_type == "news" and cfg.news_source == "bigquery":
        news_df = load_news_features_bq(
            project_id=cfg.bq_news_project_id,
            dataset_id=cfg.bq_news_dataset_id,
            table=cfg.bq_news_table,
            commodity=cfg.target_commodity,
        )

    for fold in cfg.folds:
        for window_size in cfg.window_sizes:
            print("\n" + "=" * 60)
            print(f"▶ CNN window={window_size} fold={fold}")
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

            if cfg.do_train:
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

                if cfg.do_test:
                    # Evaluate on fixed test split if available.
                    test_metrics_path = ckpt_dir / "test_metrics.json"
                    try:
                        test_ds = CNNDataset(split="test", **ds_kwargs)
                        test_loader = DataLoader(
                            test_ds,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            collate_fn=cnn_collate_fn,
                        )
                        test_loss, test_metrics, test_extra, test_pos_stats = evaluate_cnn(
                            model,
                            test_loader,
                            loss_fn,
                            device,
                            horizon_weights=horizon_weights,
                            severity_loss_weight=cfg.severity_loss_weight,
                        )
                        overall = summarize_metrics(test_metrics) if test_metrics else {}
                        test_payload = {
                            "model": "cnn",
                            "commodity": cfg.target_commodity,
                            "fold": fold,
                            "window_size": window_size,
                            "window": window_size,
                            "horizons": HORIZONS,
                            "loss": test_loss,
                            "overall": overall,
                            "summary": overall,
                            "per_horizon": test_metrics,
                            "pos_stats": test_pos_stats,
                        }
                        if test_extra:
                            test_payload["extra_metrics"] = test_extra
                        test_metrics_path.write_text(
                            json.dumps(test_payload, ensure_ascii=False, indent=2)
                        )
                        print(f"✓ Saved test metrics: {test_metrics_path}")
                    except Exception as exc:
                        print(f"⚠️  Skipping test metrics (reason: {exc})")

                print(f"✓ Saved checkpoint: {checkpoint_path}")
            elif not checkpoint_path.exists():
                print(f"⚠️  Checkpoint not found (skip inference): {checkpoint_path}")

            if cfg.do_infer and checkpoint_path.exists():
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
