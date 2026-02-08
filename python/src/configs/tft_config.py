from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TFTConfig:
    """
    TFT Training Config - Simple Version (LSTM style)
    """

    # ===== Paths =====
    data_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/datasets"
    checkpoint_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs/checkpoints"
    output_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs"
    checkpoint_layout: str = "legacy"  # legacy or simple

    # ===== Data Source =====
    data_source: str = "bigquery"  # local or bigquery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"
    split_file: str = "src/datasets/bq_splits/{commodity}_split.json"

    # ===== Data =====
    target_commodity: str = "corn"
    seq_length: int = 20
    horizons: List[int] = field(default_factory=lambda: list(range(1, 21)))
    batch_size: int = 128
    fold: List[int] = field(default_factory=lambda: [0])

    # ===== Scaling =====
    scale_x: bool = False
    scale_y: bool = True
    scaler_eps: float = 1e-8

    # ===== Model (LSTM-like settings) =====
    hidden_dim: int = 48
    num_layers: int = 2
    attention_heads: int = 6
    dropout: float = 0.2
    use_variable_selection: bool = True
    news_projection_dim: int = 48

    # ===== Loss =====
    quantile_loss: bool = False
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # ===== Training =====
    lr: float = 0.0001
    epochs: int = 300
    early_stopping_patience: int = 30
    weight_decay: float = 1e-4

    # ===== Others =====
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

    # ===== Interpretation (for TFT) =====
    compute_feature_importance: bool = True
    compute_temporal_importance: bool = True
    save_attention_weights: bool = True
    # ===== Output controls =====
    save_train_visualizations: bool = True
    save_val_predictions: bool = True


@dataclass
class TFTInferenceConfig:
    """
    TFT Inference Config
    """

    # ===== Paths =====
    data_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/datasets"
    checkpoint_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs/checkpoints"
    output_root: Optional[str] = None

    # ===== Data Source =====
    data_source: str = "bigquery"  # local or bigquery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"
    split_file: str = "src/datasets/bq_splits/{commodity}_split.json"

    # ===== Data =====
    target_commodity: str = "corn"
    seq_length: int = 20
    horizons: List[int] = field(default_factory=lambda: list(range(1, 21)))
    fold: int = 0
    split: str = "test"  # test or val

    # ===== Inference =====
    exp_name: str = ""
    checkpoint_path: Optional[str] = None
    include_targets: bool = True
    scale_x: bool = True
    scale_y: bool = True
    use_variable_selection: bool = True
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # ===== Interpretation Outputs =====
    save_importance: bool = True
    importance_groups: List[int] = field(default_factory=lambda: [5, 10, 20])
    importance_top_k: int = 20
    save_importance_images: bool = True
    save_prediction_plot: bool = True

    # ===== Runtime =====
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda"
