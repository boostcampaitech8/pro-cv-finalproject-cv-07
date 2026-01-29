from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    """
    TFT Training Config - Simple Version (LSTM style)
    """
    
    # ===== Paths =====
    data_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/datasets"
    checkpoint_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs/checkpoints"
    output_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs"
    
    # ===== Data =====
    target_commodity: str = "corn"
    seq_length: int = 20
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    batch_size: int = 64
    fold: List[int] = field(default_factory=lambda: [0])
    
    # ===== Model (LSTM-like settings) =====
    hidden_dim: int = 32            # LSTM과 동일
    num_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1            # 낮게
    use_variable_selection: bool = False
    
    # ===== Loss =====
    quantile_loss: bool = False     # MSE 사용! (LSTM과 동일)
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # ===== Training =====
    lr: float = 0.001               # LSTM과 동일
    epochs: int = 100
    early_stopping_patience: int = 20
    weight_decay: float = 1e-5
    
    # ===== Others =====
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

    # ===== Interpretation (for TFT) =====
    compute_feature_importance: bool = False
    compute_temporal_importance: bool = True
    save_attention_weights: bool = True