from dataclasses import dataclass, field
from typing import List
from tyro import conf

@dataclass
class TrainConfig:
    # ===== 데이터 경로 수정 (마지막 "/" 제거) =====
    data_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/datasets"
    # ===========================================
    
    # Target commodity
    target_commodity: str = "corn"  # "corn", "soybean", "wheat"
    
    # TFT specific
    use_variable_selection: bool = False  # 일단 False (안정성)
    attention_heads: int = 4
    dropout: float = 0.1
    
    # Loss
    quantile_loss: bool = True
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # Ensemble
    ensemble_method: str = "best"  # "average", "weighted", "best"
    
    # Interpretation
    compute_feature_importance: bool = True
    compute_temporal_importance: bool = True
    save_attention_weights: bool = True
    interpretation_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs/interpretations"
    
    # Training
    early_stopping_patience: int = 20
    weight_decay: float = 1e-5
    device: str = "cuda"
    num_workers: int = 4
    log_interval: int = 10
    save_interval: int = 5

    use_ema_features: bool = False
    ema_spans: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    use_volatility_features: bool = False
    volatility_windows: List[int] = field(default_factory=lambda: [7, 14, 21])
    use_news_count_features: bool = False
    use_news_imformation_features: bool = False
    
    batch_size: int = 64
    seq_length: int = 20
    fold: List[int] = field(default_factory=lambda: [0])  # 기본값을 fold 0으로 (테스트용)
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    hidden_dim: int = 32
    num_layers: int = 2

    lr: float = 0.001
    epochs: int = 100

    checkpoint_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs/checkpoints"
    output_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs"

    seed: int = 42
    ensemble: bool = False
