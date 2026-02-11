from dataclasses import dataclass, field
from typing import List


@dataclass
class CNNBatchConfig:
    # ===== Data =====
    data_dir: str = "src/datasets/local_bq_like/corn"
    split_file: str = "corn_split.json"
    target_commodity: str = "corn"
    folds: List[int] = field(default_factory=lambda: [0])
    data_source: str = "bigquery"  # local or bigquery
    bq_project_id: str = "esoteric-buffer-485608-g5"
    bq_dataset_id: str = "final_proj"
    bq_train_table: str = "train_price"
    bq_inference_table: str = "inference_price"
    bq_test_table: str = "test_price"
    image_source: str = "gcs"  # local or gcs
    gcs_bucket: str = "boostcamp-final-proj"
    gcs_prefix_template: str = "{symbol}/window_{window}_ohlc"
    news_source: str = "bigquery"  # csv or bigquery
    bq_news_project_id: str = "gcp-practice-484218"
    bq_news_dataset_id: str = "news_data"
    bq_news_table: str = "daily_summary"

    # ===== Windows =====
    window_sizes: List[int] = field(default_factory=lambda: [5, 20, 60])

    # ===== Model =====
    image_mode: str = "candle"
    backbone: str = "convnext_tiny"
    use_aux: bool = True
    aux_type: str = "news"
    fusion: str = "gated"

    # ===== Training =====
    loss: str = "smooth_l1"
    batch_size: int = 32
    lr: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    epochs: int = 150
    early_stop_patience: int = 30
    early_stop_metric: str = "auprc"
    min_epochs: int = 30
    freeze_backbone_epochs: int = 30
    severity_loss_weight: float = 1.0
    horizon_weights: List[float] = field(default_factory=lambda: [1.0] * 20)
    save_train_log: bool = False

    # ===== Outputs =====
    output_tag: str = "eval"
    checkpoint_root: str = "src/outputs/checkpoints/{commodity}_{date}_cnn{tag}"
    prediction_root: str = "src/outputs/predictions/{commodity}_{date}_cnn{tag}"
    date_tag: str = ""  # optional override
    exp_name: str = ""  # optional metadata tag

    # ===== Inference =====
    write_json: bool = False
    all_dates: bool = False
    inference_split: str = "infer"
    do_test: bool = False
    do_train: bool = True
    do_infer: bool = True
