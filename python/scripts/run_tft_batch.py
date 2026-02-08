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
from typing import List, Optional
import csv
import json
import sys

import tyro

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.configs.train_config import TrainConfig
from src.engine.inference_tft import run_inference_tft

# Import script entrypoints as functions
from scripts.train_tft import main as train_main
from scripts.test_tft_metrics_json import main as test_metrics_main


@dataclass
class BatchConfig:
    data_dir: str = "src/datasets/local_bq_like/corn"
    target_commodity: str = "corn"
    seq_lengths: List[int] = field(default_factory=lambda: [5, 20, 60])
    horizons: List[int] = field(default_factory=lambda: list(range(1, 21)))
    split_file: str = "src/datasets/bq_splits/{commodity}_split.json"
    fold: List[int] = field(default_factory=lambda: [0])

    # Training
    epochs: int = 300
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda"
    scale_x: bool = False
    scale_y: bool = True
    quantile_loss: bool = False
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    use_variable_selection: bool = True

    # Control flags
    do_train: bool = True
    do_test: bool = False
    do_infer: bool = True

    # Inference outputs
    save_importance_images: bool = True
    save_prediction_plot: bool = True

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


def _collect_predictions_json(output_dir: Path, horizons: List[int]) -> List[dict]:
    rows = []
    for json_path in sorted(output_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        meta = payload.get("meta", {})
        preds_dict = payload.get("predictions", {})
        preds = preds_dict.get("close") or preds_dict.get("log_return", {})
        row = {
            "date": meta.get("as_of"),
            "window": meta.get("window"),
        }
        for h in horizons:
            row[f"pred_h{h}"] = preds.get(f"h{h}")
        rows.append(row)
    return rows


def _write_combined_csv(rows: List[dict], output_path: Path, horizons: List[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["date", "window"] + [f"pred_h{h}" for h in horizons]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

        if cfg.do_test:
            test_metrics_main(train_cfg)

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
                data_source=cfg.data_source,
                bq_project_id=cfg.bq_project_id,
                bq_dataset_id=cfg.bq_dataset_id,
                bq_train_table=cfg.bq_train_table,
                bq_inference_table=cfg.bq_inference_table,
                split="inference",
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                device=cfg.device,
                use_variable_selection=cfg.use_variable_selection,
                quantiles=cfg.quantiles if cfg.quantile_loss else None,
                include_targets=False,
                scale_x=cfg.scale_x,
                scale_y=cfg.scale_y,
                save_importance=cfg.save_infer_interpretations,
                importance_groups=[5, 10, 20],
                importance_top_k=20,
                save_importance_images=cfg.save_infer_interpretations and cfg.save_importance_images,
                save_prediction_plot=cfg.save_prediction_plot,
                interpretations_use_fold_dir=False,
            )

            combined_rows.extend(_collect_predictions_json(output_dir, cfg.horizons))

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
        _write_combined_csv(combined_rows, out_path, cfg.horizons)
        print(f"\n✓ Combined predictions saved: {out_path}")


if __name__ == "__main__":
    main(tyro.cli(BatchConfig))
    data_source: str = "bigquery"  # local or bigquery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"
