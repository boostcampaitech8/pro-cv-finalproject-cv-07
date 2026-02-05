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
    batch_size: int = 128
    fold: List[int] = field(default_factory=lambda: [0])

    # ===== Scaling =====
    scale_x: bool = False
    scale_y: bool = True
    scaler_eps: float = 1e-8
    
    # ===== Model (LSTM-like settings) =====
    hidden_dim: int = 32    # 32가 base -> 64로 1차 수정 -> 다시 32로 수정 -> 48
    num_layers: int = 2
    attention_heads: int = 4    # 4이 base -> 8로 1차 수정 -> 다시 4로 수정 -> 6
    dropout: float = 0.1    # 0.1이 base -> 0.2로 1차 수정 -> 다시 0.1로 수정 -> news_projection_dim을 32에서 64로 바꾸면서 0.15로 수정 -> 0.2
    use_variable_selection: bool = True
    news_projection_dim: int = 32   #원래 32로 하는데 news_proj_17이 너무 중요도가 높아서 64로 변경 -> 중간 값 48로 변경
    
    # ===== Loss =====
    quantile_loss: bool = False
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # ===== Training =====
    lr: float = 0.0001
    epochs: int = 300
    early_stopping_patience: int = 30
    weight_decay: float = 1e-5  # 1e-5가 base -> 1e-4로 1차 수정 -> 다시 1e-5로 수정 -> 1e-4
    
    # ===== Others =====
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

    # ===== Interpretation (for TFT) =====
    compute_feature_importance: bool = True
    compute_temporal_importance: bool = True
    save_attention_weights: bool = True