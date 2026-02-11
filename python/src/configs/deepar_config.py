from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepARConfig:
    # ===== Data =====
    data_dir: str = "src/datasets/local_bq_like/corn"
    target_commodity: str = "corn"
    split_file: str = "src/datasets/local_bq_like/corn/rolling_fold_3m_corn.json"
    data_source: str = "local"  # local or bigquery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"
    future_price_file: str = "src/datasets/{commodity}_future_price.csv"
    do_test: bool = True

    # ===== Windows =====
    seq_lengths: List[int] = field(default_factory=lambda: [5, 20, 60])
    prediction_length: int = 20
    horizons: List[int] = field(default_factory=lambda: list(range(1, 21)))

    # ===== Training =====
    epochs: int = 150
    fold: List[int] = field(default_factory=lambda: [0])
    num_samples: int = 200
    seed: int = 42

    # ===== Early Stopping =====
    early_stop: bool = True
    patience: int = 10
    min_delta: float = 0.0

    # ===== Quantiles =====
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    # ===== Output Roots =====
    output_tag: str = ""  # e.g. "eval" to write ..._deepar_eval
    checkpoint_root: str = "src/outputs/checkpoints/{commodity}_{date}_deepar{tag}"
    prediction_root: str = "src/outputs/predictions/{commodity}_{date}_deepar{tag}"
    combined_output: str = "src/outputs/predictions/{commodity}_{date}_deepar{tag}/deepar_predictions.csv"
