"""
TFT Inference Script (Unified Output)

Example:
python scripts/infer_tft.py \
  --target_commodity corn \
  --fold 0 \
  --seq_length 20 \
  --horizons 1 5 10 20
"""

import tyro

from src.configs.tft_config import TFTInferenceConfig
from src.engine.inference_tft import run_inference_tft


def main(cfg: TFTInferenceConfig) -> None:
    exp_name = cfg.exp_name or None
    checkpoint_path = cfg.checkpoint_path or None
    output_root = cfg.output_root or None

    output_dir = run_inference_tft(
        commodity=cfg.target_commodity,
        fold=cfg.fold,
        seq_length=cfg.seq_length,
        horizons=cfg.horizons,
        exp_name=exp_name,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        data_dir=cfg.data_dir,
        data_source=cfg.data_source,
        bq_project_id=cfg.bq_project_id,
        bq_dataset_id=cfg.bq_dataset_id,
        bq_train_table=cfg.bq_train_table,
        bq_inference_table=cfg.bq_inference_table,
        split=cfg.split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
        seed=cfg.seed,
        use_variable_selection=cfg.use_variable_selection,
        quantiles=cfg.quantiles,
        include_targets=cfg.include_targets,
        scale_x=cfg.scale_x,
        scale_y=cfg.scale_y,
        save_importance=cfg.save_importance,
        importance_groups=cfg.importance_groups,
        importance_top_k=cfg.importance_top_k,
        save_importance_images=cfg.save_importance_images,
        save_prediction_plot=cfg.save_prediction_plot,
    )

    print(f"Unified TFT outputs saved to: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(TFTInferenceConfig))
