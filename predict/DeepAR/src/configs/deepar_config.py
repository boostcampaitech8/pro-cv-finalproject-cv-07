from dataclasses import dataclass, field
from typing import List

from shared.configs.config import BaseDataConfig, HorizonConfig, WindowConfig


@dataclass
class DeepARConfig(BaseDataConfig, HorizonConfig, WindowConfig):
    # ===== Data =====
    future_price_file: str = "shared/src/datasets/{commodity}_future_price.csv"
    do_test: bool = False

    # ===== Windows =====
    prediction_length: int = 20

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
    checkpoint_root: str = "outputs/checkpoints/{date}_{commodity}_deepar{tag}"
    prediction_root: str = "outputs/predictions/{date}_{commodity}_deepar{tag}"
    combined_output: str = "outputs/predictions/{date}_{commodity}_deepar{tag}/deepar_predictions.csv"


@dataclass
class DeepARInferenceConfig(BaseDataConfig, HorizonConfig, WindowConfig):
    # ===== Data =====
    future_price_file: str = "shared/src/datasets/{commodity}_future_price.csv"

    # ===== Windows =====
    prediction_length: int = 20

    # ===== Inference =====
    num_samples: int = 200
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    # ===== Output Roots =====
    output_tag: str = ""  # e.g. "eval" to write ..._deepar_eval
    checkpoint_root: str = "outputs/checkpoints/{date}_{commodity}_deepar{tag}"
    prediction_root: str = "outputs/predictions/{date}_{commodity}_deepar{tag}"
    combined_output: str = "outputs/predictions/{date}_{commodity}_deepar{tag}/deepar_predictions.csv"
