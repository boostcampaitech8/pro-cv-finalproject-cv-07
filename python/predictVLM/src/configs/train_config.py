from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    data_dir: str = "./shared/datasets"

    seed: int = 42