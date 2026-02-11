"""
TFT Test Metrics (JSON-only)

Usage:
python scripts/test_tft_metrics_json.py \
  --target_commodity corn \
  --data_dir src/datasets/local_bq_like/corn \
  --split_file rolling_fold_2m_corn.json \
  --fold 0
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

sys.path.append(str(Path(__file__).parent.parent))

from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset_tft import TFTDataLoader
from src.data.bigquery_loader import load_price_table, load_news_features_bq
from src.models.TFT import TemporalFusionTransformer


def _compute_metrics_per_horizon(predictions: np.ndarray, targets: np.ndarray, horizons):
    metrics = {}
    for h_idx, horizon in enumerate(horizons):
        pred_h = predictions[:, h_idx]
        true_h = targets[:, h_idx]

        mae = float(np.mean(np.abs(pred_h - true_h)))
        rmse = float(np.sqrt(np.mean((pred_h - true_h) ** 2)))
        epsilon = 1e-8
        mape = float(np.mean(np.abs((true_h - pred_h) / (np.abs(true_h) + epsilon))) * 100.0)

        ss_res = float(np.sum((true_h - pred_h) ** 2))
        ss_tot = float(np.sum((true_h - np.mean(true_h)) ** 2))
        r2 = float(1 - (ss_res / (ss_tot + epsilon)))

        da = float(np.mean((pred_h > 0) == (true_h > 0)) * 100.0)

        metrics[str(horizon)] = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "DA": da,
        }
    return metrics


def _summarize_metrics(per_horizon: dict):
    keys = ["MAE", "RMSE", "MAPE", "R2", "DA"]
    summary = {}
    for k in keys:
        vals = [v[k] for v in per_horizon.values() if k in v]
        summary[k] = float(np.mean(vals)) if vals else None
    return summary


def _run_test(model, test_loader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            outputs = model(x_test, return_attention=False)
            predictions = outputs["predictions"]

            if predictions.ndim == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[:, :, median_idx]

            preds.append(predictions.cpu())
            trues.append(y_test)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    return preds, trues


def main(config: TrainConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")

    commodity = config.target_commodity

    price_file = f"preprocessing/{commodity}_feature_engineering.csv"
    news_file = "news_features.csv"

    price_path = os.path.join(config.data_dir, price_file)
    news_path = os.path.join(config.data_dir, news_file)
    news_source = news_path

    split_file = config.split_file
    if "{commodity}" in split_file:
        split_file = split_file.format(commodity=commodity)
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = Path(config.data_dir) / split_file

    price_source = price_path
    if config.data_source == "bigquery":
        price_source = load_price_table(
            project_id=config.bq_project_id,
            dataset_id=config.bq_dataset_id,
            table=config.bq_train_table,
            commodity=commodity,
        )
    else:
        if not os.path.exists(price_path):
            print(f"{price_path} 파일 존재하지 않음")
            return
    if getattr(config, "news_source", "csv") == "bigquery":
        news_source = load_news_features_bq(
            project_id=config.bq_news_project_id,
            dataset_id=config.bq_news_dataset_id,
            table=config.bq_news_table,
            commodity=commodity,
        )

    data_loader = TFTDataLoader(
        price_data_path=price_source,
        news_data_path=news_source,
        split_file=str(split_path),
        seq_length=config.seq_length,
        horizons=config.horizons,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    for fold in config.fold:
        if getattr(config, "checkpoint_layout", "legacy") == "simple":
            fold_dir = Path(config.checkpoint_dir)
            if len(config.fold) > 1:
                fold_dir = fold_dir / f"fold_{fold}"
        else:
            h_tag = "h" + "-".join(map(str, config.horizons))
            fold_dir = Path(config.checkpoint_dir) / f"TFT_{commodity}_fold{fold}_{h_tag}"
        checkpoint_path = fold_dir / "best_model.pt"
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        test_dates, test_loader = data_loader.get_test_loader(
            fold,
            scale_x=config.scale_x,
            scale_y=config.scale_y,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_cfg = checkpoint.get("config", {})
        model = TemporalFusionTransformer(
            num_features=data_loader.X.shape[-1],
            num_horizons=len(config.horizons),
            hidden_dim=model_cfg.get("hidden_dim", config.hidden_dim),
            lstm_layers=model_cfg.get("num_layers", config.num_layers),
            attention_heads=model_cfg.get("attention_heads", config.attention_heads),
            dropout=model_cfg.get("dropout", config.dropout),
            use_variable_selection=model_cfg.get("use_variable_selection", config.use_variable_selection),
            quantiles=config.quantiles if config.quantile_loss else None,
            news_projection_dim=model_cfg.get("news_projection_dim", getattr(config, "news_projection_dim", 32)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        preds, trues = _run_test(model, test_loader, device)

        scaler_path = fold_dir / "scaler.npz"
        if scaler_path.exists():
            sc = np.load(scaler_path, allow_pickle=True)
            if bool(sc["scale_y"]):
                y_mean = sc["y_mean"].astype(np.float32)
                y_std = sc["y_std"].astype(np.float32)
                preds = preds * y_std[None, :] + y_mean[None, :]
                trues = trues * y_std[None, :] + y_mean[None, :]

        per_horizon = _compute_metrics_per_horizon(preds, trues, config.horizons)
        summary = _summarize_metrics(per_horizon)

        out = {
            "model": "tft",
            "commodity": commodity,
            "fold": fold,
            "seq_length": config.seq_length,
            "horizons": config.horizons,
            "split_file": str(split_path),
            "test_range": {
                "start": str(test_dates[0])[:10] if test_dates else None,
                "end": str(test_dates[-1])[:10] if test_dates else None,
                "n": len(test_dates),
            },
            "per_horizon": per_horizon,
            "summary": summary,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        out_path = fold_dir / "test_metrics.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"✓ Saved test metrics: {out_path}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
