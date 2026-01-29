from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    name: str = "corn"
    data_dir: str = "./shared/datasets"
    
    seq_length: int = 20
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    fold: int = 7

    seed: int = 42