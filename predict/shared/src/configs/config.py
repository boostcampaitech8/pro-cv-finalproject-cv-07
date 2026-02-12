from dataclasses import dataclass, field
from typing import List


@dataclass
class BaseDataConfig:
    # ===== Data Source =====
    data_dir: str = "shared/src/datasets"
    split_file: str = "shared/src/datasets/bq_splits/{commodity}_split.json"
    target_commodity: str = "corn"
    data_source: str = "bigquery"  # local or bigquery
    bq_project_id: str = ""
    bq_dataset_id: str = ""
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"


@dataclass
class NewsConfig:
    # ===== News Source =====
    news_source: str = "bigquery"  # csv or bigquery
    bq_news_project_id: str = ""
    bq_news_dataset_id: str = ""
    bq_news_table: str = "daily_summary"


@dataclass
class HorizonConfig:
    # ===== Horizons =====
    horizons: List[int] = field(default_factory=lambda: list(range(1, 21)))


@dataclass
class WindowConfig:
    # ===== Windows =====
    seq_lengths: List[int] = field(default_factory=lambda: [5, 20, 60])
