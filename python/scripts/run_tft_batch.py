"""
Run TFT training/test/inference for multiple window sizes in one command.

Outputs:
  - best_model / metrics per window (separate checkpoint roots)
  - inference JSON + PNGs per window
  - combined CSV with predictions across windows

Example:
python scripts/run_tft_batch.py \
  --data_dir src/datasets/local_bq_like/corn \
  --target_commodity corn \
  --seq_lengths 5 20 60 \
  --split_file rolling_fold_2m_corn.json \
  --fold 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import List, Optional, Dict
import csv
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import json
import sys
import os
import warnings
import logging

import tyro

# Silence noisy third-party warnings (transformers/torch version checks)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.configs.train_config import TrainConfig
from src.engine.inference_tft import run_inference_tft

# Import script entrypoints as functions
from scripts.train_tft import main as train_main
from scripts.test_tft_metrics_future import evaluate_from_inference


@dataclass
class BatchConfig:
    data_dir: str = "src/datasets/local_bq_like/corn"
    target_commodity: str = "corn"
    seq_lengths: List[int] = field(default_factory=lambda: [5, 20, 60])
    horizons: List[int] = field(default_factory=lambda: list(range(1, 21)))
    split_file: str = "src/datasets/bq_splits/{commodity}_split.json"
    fold: List[int] = field(default_factory=lambda: [0])
    data_source: str = "local"  # local or bigquery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"
    future_price_file: str = "src/datasets/{commodity}_future_price.csv"
    eval_last_only: bool = True

    # Training
    epochs: int = 300
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    scale_x: bool = False
    scale_y: bool = True
    quantile_loss: bool = False
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    use_variable_selection: bool = True
    # If horizons are sparse (e.g., h1,h5,h10,h20), optionally interpolate
    # log_return predictions to full horizon range for output CSV.
    interpolate_horizons: bool = False
    interpolate_max_horizon: int = 20

    # Control flags
    do_train: bool = True
    do_test: bool = False
    do_infer: bool = True

    # Inference outputs
    save_importance_images: bool = True
    save_prediction_plot: bool = True
    importance_groups: List[int] = field(default_factory=lambda: [20])

    # Interpretation toggles
    save_train_interpretations: bool = False
    save_infer_interpretations: bool = True

    # Training artifact toggles
    save_train_visualizations: bool = False
    save_val_predictions: bool = False

    # Checkpoints root (per window subdir will be created)
    checkpoint_root: str = "src/outputs/checkpoints/{commodity}_{date}_tft"
    checkpoint_layout: str = "simple"  # legacy or simple

    # Prediction outputs
    prediction_root: str = "src/outputs/predictions/{commodity}_{date}_tft"
    combined_output: str = "src/outputs/predictions/{commodity}_{date}_tft/tft_predictions.csv"


def _build_train_config(cfg: BatchConfig, seq_length: int, checkpoint_dir: Path) -> TrainConfig:
    return TrainConfig(
        data_dir=cfg.data_dir,
        target_commodity=cfg.target_commodity,
        seq_length=seq_length,
        horizons=cfg.horizons,
        fold=cfg.fold,
        split_file=cfg.split_file,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
        seed=cfg.seed,
        scale_x=cfg.scale_x,
        scale_y=cfg.scale_y,
        quantile_loss=cfg.quantile_loss,
        quantiles=cfg.quantiles,
        use_variable_selection=cfg.use_variable_selection,
        data_source=cfg.data_source,
        bq_project_id=cfg.bq_project_id,
        bq_dataset_id=cfg.bq_dataset_id,
        bq_train_table=cfg.bq_train_table,
        bq_inference_table=cfg.bq_inference_table,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_layout=cfg.checkpoint_layout,
        compute_feature_importance=cfg.save_train_interpretations,
        compute_temporal_importance=cfg.save_train_interpretations,
        save_train_visualizations=cfg.save_train_visualizations,
        save_val_predictions=cfg.save_val_predictions,
    )


def _checkpoint_path(
    checkpoint_dir: Path,
    commodity: str,
    fold: int,
    horizons: List[int],
    layout: str,
    num_folds: int,
) -> Path:
    if layout == "simple":
        base = checkpoint_dir
        if num_folds > 1:
            base = base / f"fold_{fold}"
        return base / "best_model.pt"
    h_tag = "h" + "-".join(map(str, horizons))
    return checkpoint_dir / f"TFT_{commodity}_fold{fold}_{h_tag}" / "best_model.pt"


def _interpolate_log_returns(known: Dict[int, float], max_h: int) -> Dict[int, float]:
    if not known:
        return {}
    xs = sorted(known.keys())
    ys = [known[h] for h in xs]
    # Clamp outside range to endpoints (avoids extrapolation beyond known points)
    full = {}
    for h in range(1, max_h + 1):
        if h <= xs[0]:
            full[h] = ys[0]
        elif h >= xs[-1]:
            full[h] = ys[-1]
        else:
            # linear interpolation in log-return space (exponential in price)
            # find segment
            for i in range(1, len(xs)):
                if xs[i] >= h:
                    x0, x1 = xs[i - 1], xs[i]
                    y0, y1 = ys[i - 1], ys[i]
                    t = (h - x0) / (x1 - x0)
                    full[h] = y0 + t * (y1 - y0)
                    break
    return full


def _collect_predictions_json(
    output_dir: Path,
    horizons: List[int],
    *,
    interpolate: bool = False,
    max_horizon: int = 20,
) -> List[dict]:
    rows: List[dict] = []
    for json_path in sorted(output_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        meta = payload.get("meta", {})
        preds_dict = payload.get("predictions", {})
        preds = preds_dict.get("log_return", {})
        as_of = str(meta.get("as_of"))
        window = meta.get("window")
        if interpolate:
            known = {}
            for h in horizons:
                val = preds.get(f"h{h}")
                if val is None:
                    continue
                try:
                    known[h] = float(val)
                except Exception:
                    continue
            full = _interpolate_log_returns(known, max_horizon)
            use_h = sorted(full.keys())
            for h in use_h:
                rows.append(
                    {
                        "date": as_of,
                        "window": window,
                        "horizon": h,
                        "pred_value": full.get(h),
                    }
                )
            continue
        for h in horizons:
            rows.append(
                {
                    "date": as_of,
                    "window": window,
                    "horizon": h,
                    "pred_value": preds.get(f"h{h}"),
                }
            )
    return rows


def _write_combined_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["date", "window", "predict_date", "predicted_close"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_close_map(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None or "close" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "close"])
    return dict(zip(df["date"].tolist(), df["close"].tolist()))


def _build_trading_calendar(inference_path: Path, future_path: Path) -> List[str]:
    dates = []
    for p in [inference_path, future_path]:
        if p.exists():
            df = pd.read_csv(p)
            if "time" in df.columns:
                dates.extend(pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
            elif "trade_date" in df.columns:
                dates.extend(pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
            elif "date" in df.columns:
                dates.extend(pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
    dates = [d for d in dates if d and d != "NaT"]
    return sorted(set(dates))


def _next_trading_dates(
    as_of: str,
    horizons: List[int],
    trading_calendar: List[str],
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if trading_calendar and as_of in trading_calendar:
        idx = trading_calendar.index(as_of)
        future = trading_calendar[idx + 1 :]
        for h in horizons:
            if h - 1 < len(future):
                mapping[h] = future[h - 1]
    # fallback (or fill gaps): business days (skips weekends, not exchange holidays)
    base = pd.to_datetime(as_of)
    for h in horizons:
        if h not in mapping:
            mapping[h] = (base + BDay(h)).strftime("%Y-%m-%d")
    return mapping


def _infer_as_of_date(output_dir: Path) -> Optional[str]:
    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        return None
    payload = json.loads(json_files[0].read_text())
    meta = payload.get("meta", {})
    as_of = meta.get("as_of")
    if as_of:
        return str(as_of)[:10]
    return json_files[0].stem


def main(cfg: BatchConfig) -> None:
    combined_rows: List[dict] = []
    date_tag: Optional[str] = None
    temp_root: Optional[Path] = None
    temp_ckpt_root: Optional[Path] = None

    for seq_length in cfg.seq_lengths:
        print(f"\n{'='*60}")
        print(f"▶ Running window={seq_length}")
        print(f"{'='*60}")

        if date_tag is None:
            temp_ckpt_root = Path("src/outputs/checkpoints") / f"{cfg.target_commodity}_tft_tmp"
            checkpoint_dir = temp_ckpt_root / f"w{seq_length}"
        else:
            checkpoint_dir = Path(
                cfg.checkpoint_root.format(
                    commodity=cfg.target_commodity,
                    date=date_tag,
                )
            ) / f"w{seq_length}"
        train_cfg = _build_train_config(cfg, seq_length, checkpoint_dir)

        if cfg.do_train:
            train_main(train_cfg)

        if cfg.do_infer:
            fold = cfg.fold[0] if cfg.fold else 0
            ckpt_path = _checkpoint_path(
                checkpoint_dir,
                cfg.target_commodity,
                fold,
                cfg.horizons,
                cfg.checkpoint_layout,
                len(cfg.fold),
            )
            exp_name = f"tft_w{seq_length}_h{'-'.join(map(str, cfg.horizons))}"
            if date_tag is None:
                temp_root = (
                    Path("src/outputs/predictions")
                    / f"{cfg.target_commodity}_tft_tmp"
                )
                inference_root = temp_root / f"w{seq_length}" / "results"
            else:
                inference_root = (
                    Path(cfg.prediction_root.format(commodity=cfg.target_commodity, date=date_tag))
                    / f"w{seq_length}"
                    / "results"
                )

            output_dir = run_inference_tft(
                commodity=cfg.target_commodity,
                fold=fold,
                seq_length=seq_length,
                horizons=cfg.horizons,
                exp_name=exp_name,
                checkpoint_path=str(ckpt_path),
                output_root=str(inference_root),
                data_dir=cfg.data_dir,
                split_file=cfg.split_file,
                data_source=cfg.data_source,
                bq_project_id=cfg.bq_project_id,
                bq_dataset_id=cfg.bq_dataset_id,
                bq_train_table=cfg.bq_train_table,
                bq_inference_table=cfg.bq_inference_table,
                split="inference",
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                device=cfg.device,
                seed=cfg.seed,
                use_variable_selection=cfg.use_variable_selection,
                quantiles=cfg.quantiles if cfg.quantile_loss else None,
                include_targets=False,
                scale_x=cfg.scale_x,
                scale_y=cfg.scale_y,
                save_importance=cfg.save_infer_interpretations,
                importance_groups=cfg.importance_groups,
                importance_top_k=20,
                save_importance_images=cfg.save_infer_interpretations and cfg.save_importance_images,
                save_prediction_plot=cfg.save_prediction_plot,
                interpretations_use_fold_dir=False,
            )

            combined_rows.extend(
                _collect_predictions_json(
                    output_dir,
                    cfg.horizons,
                    interpolate=cfg.interpolate_horizons,
                    max_horizon=cfg.interpolate_max_horizon,
                )
            )

            if cfg.do_test:
                future_path = Path(
                    cfg.future_price_file.format(commodity=cfg.target_commodity)
                )
                inference_price = Path(cfg.data_dir) / "inference_price.csv"
                evaluate_from_inference(
                    inference_root=output_dir,
                    inference_price=inference_price,
                    future_price=future_path,
                    horizons=cfg.horizons,
                    commodity=cfg.target_commodity,
                    window=seq_length,
                    fold=fold,
                    checkpoint_dir=checkpoint_dir,
                    last_only=cfg.eval_last_only,
                    interpolate_horizons=cfg.interpolate_horizons,
                    interpolate_max_horizon=cfg.interpolate_max_horizon,
                )

            if date_tag is None:
                inferred = _infer_as_of_date(output_dir)
                if inferred:
                    date_tag = inferred
                    if temp_root is not None:
                        final_root = Path(
                            cfg.prediction_root.format(
                                commodity=cfg.target_commodity,
                                date=date_tag,
                            )
                        )
                        final_root.parent.mkdir(parents=True, exist_ok=True)
                        if final_root.exists():
                            shutil.rmtree(final_root)
                        shutil.move(str(temp_root), str(final_root))
                    if temp_ckpt_root is not None:
                        final_ckpt_root = Path(
                            cfg.checkpoint_root.format(
                                commodity=cfg.target_commodity,
                                date=date_tag,
                            )
                        )
                        final_ckpt_root.parent.mkdir(parents=True, exist_ok=True)
                        if final_ckpt_root.exists():
                            shutil.rmtree(final_ckpt_root)
                        shutil.move(str(temp_ckpt_root), str(final_ckpt_root))

    if combined_rows:
        if date_tag is None:
            out_path = Path(
                cfg.combined_output.format(
                    commodity=cfg.target_commodity,
                    date="unknown",
                )
            )
        else:
            out_path = Path(
                cfg.combined_output.format(
                    commodity=cfg.target_commodity,
                    date=date_tag,
                )
            )

        inference_price = Path(cfg.data_dir) / "inference_price.csv"
        future_price = Path(cfg.future_price_file.format(commodity=cfg.target_commodity))
        trading_calendar = _build_trading_calendar(inference_price, future_price)
        close_map = _load_close_map(inference_price)

        formatted_rows = []
        for row in combined_rows:
            as_of = str(row.get("date"))[:10]
            window = row.get("window")
            horizon = int(row.get("horizon"))
            pred_lr = row.get("pred_value")
            if pred_lr is None:
                continue
            base_close = close_map.get(as_of)
            if base_close is None or not np.isfinite(base_close):
                continue
            pred_close = float(base_close * np.exp(float(pred_lr)))
            date_map = _next_trading_dates(as_of, [horizon], trading_calendar)
            predict_date = date_map.get(horizon)
            if predict_date is None:
                continue
            formatted_rows.append(
                {
                    "date": as_of,
                    "window": window,
                    "predict_date": predict_date,
                    "predicted_close": pred_close,
                }
            )

        _write_combined_csv(formatted_rows, out_path)
        print(f"\n✓ Combined predictions saved: {out_path}")


if __name__ == "__main__":
    main(tyro.cli(BatchConfig))
