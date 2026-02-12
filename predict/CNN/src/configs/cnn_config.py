from dataclasses import dataclass, field
from typing import List

from shared.configs.config import BaseDataConfig, NewsConfig, WindowConfig


@dataclass
class CNNBatchConfig(BaseDataConfig, NewsConfig, WindowConfig):
    # ===== Data =====
    data_dir: str = "shared/src/datasets/local_bq_like/corn"
    folds: List[int] = field(default_factory=lambda: [0])
    # Test table not used (evaluation disabled)
    bq_test_table: str = ""
    image_source: str = "gcs"  # local or gcs
    gcs_bucket: str = ""
    gcs_prefix_template: str = "{symbol}/window_{window}_ohlc"

    # ===== Windows =====

    # ===== Model =====
    image_mode: str = "candle"
    backbone: str = "convnext_tiny"
    use_aux: bool = True
    aux_type: str = "news"
    fusion: str = "gated"

    # ===== Training =====
    loss: str = "smooth_l1"
    batch_size: int = 128
    lr: float = 1e-4
    num_workers: int = 16
    seed: int = 42
    epochs: int = 150
    early_stop_patience: int = 30
    early_stop_metric: str = "auprc"
    min_epochs: int = 30
    freeze_backbone_epochs: int = 30
    severity_loss_weight: float = 1.0
    horizon_weights: List[float] = field(default_factory=lambda: [1.0] * 20)
    save_train_log: bool = False

    # ===== Outputs =====
    output_tag: str = ""
    checkpoint_root: str = "outputs/checkpoints/{date}_{commodity}_cnn{tag}"
    prediction_root: str = "outputs/predictions/{date}_{commodity}_cnn{tag}"
    date_tag: str = ""  # optional override
    exp_name: str = ""  # optional metadata tag

    # ===== Inference =====
    write_json: bool = False
    all_dates: bool = False
    inference_split: str = "infer"
    do_test: bool = False
    do_train: bool = True
    do_infer: bool = True
