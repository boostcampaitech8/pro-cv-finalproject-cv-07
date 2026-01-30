from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    name: str = "corn"
    data_dir: str = "./shared/datasets"
    image_dir: str = "./shared/datasets/images/window_20_ema5_20"
    
    seq_length: int = 20
    ema_spans: List[int] = field(default_factory=lambda: [5, 20])
    horizons: int = 20
    fold: int = 7
    
    checkpoint_dir: str = "./predictVLM/src/outputs/checkpoints"
    output_dir: str = "./predictVLM/src/outputs"
    
    best_epoch: int = -1

    seed: int = 42