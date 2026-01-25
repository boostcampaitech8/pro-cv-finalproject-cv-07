from dataclasses import dataclass, field
from typing import List
from tyro import conf

@dataclass
class TrainConfig:
    data_dir: str = "./src/datasets"
    
    use_ema_features: bool = False
    ema_spans: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    use_volatility_features: bool = False
    volatility_windows: List[int] = field(default_factory=lambda: [7, 14, 21])
    use_news_count_features: bool = False
    use_news_imformation_features: bool = False
    
    batch_size: int = 64
    seq_length: int = 20
    fold: List[int] = field(default_factory=lambda: [7])
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    hidden_dim: int = 32
    num_layers: int = 2

    lr: float = 0.001
    epochs: int = 100

    checkpoint_dir: str = "./src/outputs/checkpoints"
    output_dir: str = "./src/outputs"

    seed: int = 42
    ensemble: bool = False