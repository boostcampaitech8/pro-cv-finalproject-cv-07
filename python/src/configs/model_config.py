from dataclasses import dataclass, field
from typing import List
from tyro import conf


@dataclass
class TrainConfig:
    """TFT 모델 학습을 위한 설정"""
    
    # ============ Data Configuration ============
    data_dir: str = "./src/datasets"
    
    # Feature Engineering
    use_ema_features: bool = False
    ema_spans: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    use_volatility_features: bool = False
    volatility_windows: List[int] = field(default_factory=lambda: [7, 14, 21])
    use_news_count_features: bool = True  # 뉴스 개수 feature
    use_news_embedding_features: bool = True  # 뉴스 embedding feature
    use_multi_commodity_features: bool = True  # 다른 상품 가격 정보
    
    # Target commodity
    target_commodity: str = "corn"  # "corn", "soybean", "wheat"
    
    # ============ Data Split Configuration ============
    batch_size: int = 64
    seq_length: int = 20  # window size (lookback period)
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])  # multi-horizon targets
    fold: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])  # 사용할 fold 리스트
    
    # ============ TFT Model Configuration ============
    # Hidden dimensions
    hidden_dim: int = 64  # LSTM보다 크게 설정 (TFT는 더 복잡)
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    
    # Variable selection network
    use_variable_selection: bool = True
    
    # Static variables (사용 안 함 - 시계열만)
    static_variables: List[str] = field(default_factory=list)
    
    # ============ Training Configuration ============
    lr: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 20
    weight_decay: float = 1e-5
    
    # Loss function
    quantile_loss: bool = True  # Quantile loss 사용 (불확실성 측정)
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # ============ Ensemble Configuration ============
    ensemble: bool = False
    ensemble_method: str = "average"  # "average", "weighted", "best"
    # best: 가장 성능 좋은 fold만 사용
    # average: 모든 fold 평균
    # weighted: 성능에 따라 가중 평균
    
    # ============ Interpretation Configuration ============
    compute_feature_importance: bool = True  # Variable importance 계산
    compute_temporal_importance: bool = True  # Attention weights 저장
    save_attention_weights: bool = True  # Attention 시각화용 저장
    
    # ============ Output Configuration ============
    checkpoint_dir: str = "./src/outputs/checkpoints"
    output_dir: str = "./src/outputs"
    interpretation_dir: str = "./src/outputs/interpretations"
    
    # ============ Others ============
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4
    
    # Logging
    log_interval: int = 10  # 몇 step마다 로그 출력
    save_interval: int = 5  # 몇 epoch마다 체크포인트 저장
    
    def __post_init__(self):
        """설정 검증"""
        assert self.target_commodity in ["corn", "soybean", "wheat"], \
            f"target_commodity must be one of ['corn', 'soybean', 'wheat'], got {self.target_commodity}"
        
        assert len(self.horizons) > 0, "At least one horizon must be specified"
        
        assert self.ensemble_method in ["average", "weighted", "best"], \
            f"ensemble_method must be one of ['average', 'weighted', 'best'], got {self.ensemble_method}"
        
        # fold 개수 확인
        if len(self.fold) == 0:
            self.fold = list(range(8))  # 기본값: 모든 fold 사용


@dataclass  
class ModelConfig:
    """TFT 모델 아키텍처 상세 설정"""
    
    # Input dimensions (자동으로 계산됨)
    num_static_variables: int = 0
    num_time_varying_categorical: int = 0
    num_time_varying_real: int = 50  # placeholder, 실제로는 데이터에서 계산
    
    # Embedding dimensions
    categorical_embedding_sizes: dict = field(default_factory=dict)
    
    # LSTM
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    
    # Attention
    attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Variable selection
    hidden_continuous_size: int = 32
    
    # GRN (Gated Residual Network)
    grn_hidden_size: int = 64
    
    # Output
    output_quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # Regularization
    dropout: float = 0.1
    
    def update_from_train_config(self, train_config: TrainConfig):
        """TrainConfig에서 값 업데이트"""
        self.lstm_hidden_size = train_config.hidden_dim
        self.lstm_layers = train_config.lstm_layers
        self.attention_heads = train_config.attention_heads
        self.dropout = train_config.dropout
        
        if train_config.quantile_loss:
            self.output_quantiles = train_config.quantiles
