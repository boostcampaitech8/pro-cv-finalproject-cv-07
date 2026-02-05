from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfigCNN:
    """
    Basic training config for the CNN anomaly subsystem.
    """

    # ===== Paths =====
    data_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/datasets"
    image_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/datasets/images"
    checkpoint_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs/checkpoints"
    output_dir: str = "/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs"

    # ===== Data =====
    image_size: int = 224
    in_channels: int = 3
    num_classes: int = 2
    batch_size: int = 32
    train_split: float = 0.8
    valid_split: float = 0.1
    shuffle: bool = True
    class_names: List[str] = field(default_factory=lambda: ["normal", "anomaly"])

    # ===== Model =====
    backbone: str = "simple_cnn"
    dropout: float = 0.1

    # ===== Training =====
    lr: float = 1e-3
    epochs: int = 50
    weight_decay: float = 1e-4

    # ===== Others =====
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
