from dataclasses import dataclass, field
from typing import List, Optional

from shared.configs.config import BaseDataConfig, HorizonConfig, NewsConfig, WindowConfig


@dataclass
class TFTConfig(BaseDataConfig, NewsConfig, HorizonConfig, WindowConfig):
    """
    TFT Training Config
    """

    # ===== Paths =====
    checkpoint_dir: str = "outputs/checkpoints"
    output_dir: str = "outputs"
    checkpoint_layout: str = "simple"  # legacy or simple

    # ===== Data =====
    seq_length: int = 20
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
    compute_feature_importance: bool = False
    compute_temporal_importance: bool = False
    save_attention_weights: bool = True
    # ===== Output controls =====
    save_train_visualizations: bool = False
    save_val_predictions: bool = False


@dataclass
class TFTInferenceConfig(BaseDataConfig, NewsConfig, HorizonConfig, WindowConfig):
    """
    TFT Inference Config
    """

    # ===== Paths =====
    checkpoint_dir: str = "outputs/checkpoints"
    output_root: Optional[str] = None

    # ===== Data =====
    seq_length: int = 20
    fold: int = 0
    split: str = "inference"  # test, val, or inference

    # ===== Inference =====
    exp_name: str = ""
    checkpoint_path: Optional[str] = None
    include_targets: bool = False
    scale_x: bool = True
    scale_y: bool = True
    use_variable_selection: bool = True
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # ===== Interpretation Outputs =====
    save_importance: bool = True
    importance_groups: List[int] = field(default_factory=list)
    importance_top_k: int = 20
    save_importance_images: bool = True
    save_prediction_plot: bool = True

    # ===== Combined CSV (optional) =====
    write_combined_csv: bool = True
    combined_csv_root: Optional[str] = None

    # ===== Runtime =====
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
