# TFT Stock Price Prediction

Temporal Fusion Transformer (TFT)를 활용한 Multi-Horizon 주식 가격 예측 프로젝트

## 주요 특징

### 1. Multi-Horizon Prediction
- 단일 모델로 여러 미래 시점 동시 예측 (t+1, t+5, t+10, t+20)
- Quantile regression을 통한 불확실성 추정 가능

### 2. 해석 가능성 (Interpretability)
- **Variable Importance**: 어떤 feature가 예측에 중요한지 분석
- **Temporal Importance**: 어떤 시점이 예측에 중요한지 분석
- 뉴스 embedding의 기여도 분석

### 3. Rolling Fold Cross-Validation
- 8개의 fold를 사용한 robust한 평가
- Fold별 학습 후 앙상블 또는 best fold 선택

### 4. Multi-Commodity Support
- 옥수수, 대두, 밀 등 여러 상품 가격 예측
- 상품 간 상관관계를 활용한 feature 구성

## 프로젝트 구조

```
pro-csv-fianlproject-cv-07  # git repo TFT brunch
└── Python/
    ├── __init__.py
    ├── EDA/
    │   └── EDA.ipynb
    ├── scripts/
    │   ├── preprocessing.py          # 데이터 전처리
    │   ├── train_tft.py             # TFT 학습 메인 스크립트
    │   ├── test_tft.py                  # 테스트/평가
    │   ├── view_interpretation.py
    │   └── train_deepar.py
    ├── src/
    │   ├── configs/
    │   │   ├── __init__.py
    │   │   ├── train_config.py      # 학습 설정
    │   │   └── model_config.py      # 모델 아키텍처 설정
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── dataset.py           # PyTorch Dataset (LSTM용)
    │   │   ├── dataset_tft.py       # PyTorch Dataset (TFT용)
    │   │   ├── news_preprocessing.py # 뉴스 데이터 전처리
    │   │   ├── feature_engineering.py
    │   │   ├── preprocessing.py
    │   │   └── postprocessing.py
    │   ├── datasets/
    │   │   ├── preprocessing/
    │   │   │   ├── corn_feature_engineering.csv
    │   │   │   ├── soybean_feature_engineering.csv
    │   │   │   ├── wheat_feature_engineering.csv
    │   │   │   ├── gold_feature_engineering.csv
    │   │   │   ├── silver_feature_engineering.csv
    │   │   │   └── wheat_feature_engineering.csv
    │   │   ├── corn_future_price.csv
    │   │   ├── soybean_future_price.csv
    │   │   ├── wheat_future_price.csv
    │   │   ├── cooper_future_price.csv
    │   │   ├── gold_future_price.csv
    │   │   ├── silver_future_price.csv
    │   │   ├── news_articles_resources.csv
    │   │   ├── news_features.csv
    │   │   └── rolling_fold.json
    │   ├── engine/
    │   │   ├── __init__.py
    │   │   ├── trainer.py           # LSTM Trainer
    │   │   ├── trainer_tft.py       # TFT Trainer
    │   │   └── inference.py
    │   ├── metrics/
    │   │   ├── __init__.py
    │   │   └── metrics.py
    │   ├── models/
    │   │   ├── LSTM.py              # 기존 LSTM 모델
    │   │   ├── TFT.py               # TFT 모델
    │   │   └── ensemble.py          # Fold 앙상블
    │   ├── interpretation/
    │   │   ├── interpretation.py    # 중요도 분석
    │   │   └── visualizer.py
    │   ├── outputs/
    │   │   ├── checkpoints/         # 학습된 모델
    │   │   ├── predictions/
    │   │   │   ├── corn_fold_0_test_metrics.json         
    │   │   │   ├── corn_fold_0_test_predictions.csv
    │   │   │   └── corn_predictions.csv
    │   │   ├── visualizations/      
    │   │   │       ├── fold_0_loss_curve.png        
    │   │   │       ├── fold_0_test_all_horizons.png
    │   │   │       ├── fold_0_test_h1_horizons.png
    │   │   │       ├── fold_0_test_h5_horizons.png
    │   │   │       ├── fold_0_test_h10_horizons.png
    │   │   │       └── fold_0_test_h20_horizons.png
    │   │   └── interpretations/     # 중요도 분석 결과
    │   └── utils/
    │       ├── __init__.py
    │       ├── set_seed.py
    │       └── visualization.py
    └── requirements.txt
```

## 설치 방법

### 1. 환경 설정
```bash
# Python 3.8+ 권장
conda create -n tft_stock python=3.9
conda activate tft_stock
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
tyro>=0.5.0
```

## 사용 방법

### 1. 데이터 전처리

#### 뉴스 데이터 전처리
```bash
cd src/data
python news_preprocessing.py
```

이 스크립트는:
- 뉴스 기사를 날짜별로 집계
- 뉴스 embedding을 mean pooling
- 뉴스 개수 계산
- 가격 데이터와 병합

### 2. 모델 학습

#### 기본 학습 (모든 fold)
```bash
cd scripts
python train_tft.py \
    --target_commodity corn \
    --seq_length 20 \
    --horizons 1 5 10 20 \
    --fold 0 1 2 3 4 5 6 7 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 64 \
    --lr 0.001
```

#### 특정 fold만 학습
```bash
python train_tft.py \
    --target_commodity corn \
    --fold 0 1 2  # fold 0, 1, 2만 학습
```

#### 뉴스 feature 제외
```bash
python train_tft.py \
    --target_commodity corn \
    --use_news_embedding_features False \
    --use_news_count_features False
```

#### Ensemble 설정
```bash
# 모든 fold 평균
python train_tft.py \
    --ensemble True \
    --ensemble_method average

# 가중 평균 (성능 기반)
python train_tft.py \
    --ensemble True \
    --ensemble_method weighted

# Best fold만 사용
python train_tft.py \
    --ensemble False  # 자동으로 best fold 선택
```

### 3. 해석 가능성 분석

학습 시 자동으로 수행되지만, 별도로 분석하려면:

```bash
cd scripts
python interpret.py \
    --checkpoint_dir ../src/outputs/checkpoints \
    --interpretation_dir ../src/outputs/interpretations \
    --fold_index 0  # 분석할 fold
```

### 4. 결과 확인

#### 예측 결과
```
src/outputs/predictions/corn_predictions.csv
```

#### 중요도 분석 결과
```
src/outputs/interpretations/
├── fold_0_interpretation.npz
├── corn_visualizations/
│   ├── feature_importance.png
│   ├── temporal_feature_heatmap.png
│   ├── attention_heatmap.png
│   └── temporal_importance.png
└── fold_summary.json
```

## 주요 설정 옵션

### TrainConfig 주요 파라미터

```python
# 데이터 관련
target_commodity: str = "corn"  # "corn", "soybean", "wheat"
seq_length: int = 20  # window size
horizons: List[int] = [1, 5, 10, 20]  # 예측할 미래 시점

# Feature 관련
use_news_embedding_features: bool = True
use_news_count_features: bool = True
use_multi_commodity_features: bool = True  # 다른 상품 정보 포함

# 모델 관련
hidden_dim: int = 64
lstm_layers: int = 2
attention_heads: int = 4
dropout: float = 0.1
use_variable_selection: bool = True

# 학습 관련
batch_size: int = 64
lr: float = 0.001
epochs: int = 100
early_stopping_patience: int = 20

# Quantile Loss
quantile_loss: bool = True
quantiles: List[float] = [0.1, 0.5, 0.9]

# Ensemble
ensemble: bool = False
ensemble_method: str = "best"  # "average", "weighted", "best"

# 해석 가능성
compute_feature_importance: bool = True
compute_temporal_importance: bool = True
save_attention_weights: bool = True
```

## 모델 아키텍처

### TFT 구조
```
Input [batch, seq_len, features]
    ↓
Variable Selection Network (학습 가능한 feature 선택)
    ↓
LSTM Encoder (시계열 패턴 학습)
    ↓
Multi-Head Attention (시점별 중요도)
    ↓
Gated Residual Network
    ↓
Multi-Horizon Output [batch, num_horizons]
```

### 주요 컴포넌트

1. **Variable Selection Network**
   - 각 feature의 중요도를 학습
   - 불필요한 feature 자동 제거

2. **LSTM Encoder**
   - 시계열 패턴 포착
   - Hidden state를 다음 단계로 전달

3. **Multi-Head Attention**
   - 어떤 시점이 중요한지 학습
   - Interpretability 제공

4. **Quantile Regression**
   - 예측 불확실성 추정
   - 신뢰구간 제공

## 해석 결과 예시

### Variable Importance
```
Top 10 Most Important Features:
1. Close (0.0842)
2. Volume (0.0635)
3. news_count (0.0521)
4. news_emb_0 (0.0412)
5. soybean_Close (0.0389)
...
```

### News Impact Analysis
```
News Feature Analysis:
- Number of news features: 769
- News importance ratio: 23.45%
- Top news feature: news_emb_0
- Interpretation: 뉴스가 예측에 유의미한 영향을 미침
```

### Temporal Importance
```
Key Timesteps (most attended):
1. t-1 (가장 최근)
2. t-5
3. t-10
...
```

## 성능 평가

### Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Horizon별 개별 평가

### Fold Summary 예시
```json
{
  "fold_0": {
    "best_epoch": 45,
    "best_valid_loss": 0.0234,
    "final_train_loss": 0.0189,
    "final_valid_loss": 0.0241
  },
  ...
  "best_fold": 3
}
```

## 다음 단계 (VLM 연동)

학습된 모델의 예측 결과와 해석 데이터를 VLM에 입력:

1. **예측 결과**: `predictions.csv`
2. **변수 중요도**: `feature_importance.png`
3. **시점 중요도**: `temporal_importance.png`
4. **Attention 히트맵**: `attention_heatmap.png`

VLM이 다음을 수행:
- 예측 이유 설명
- 주요 feature 강조
- 시점별 영향 설명
- 뉴스 영향 요약

## 문제 해결

### CUDA Out of Memory
```bash
# Batch size 줄이기
python train_tft.py --batch_size 32

# Hidden dimension 줄이기
python train_tft.py --hidden_dim 32
```

### 학습이 너무 느림
```bash
# Epoch 줄이기
python train_tft.py --epochs 50

# Early stopping patience 조정
python train_tft.py --early_stopping_patience 10
```

### 특정 fold 건너뛰기
```bash
# Fold 3, 5만 제외하고 학습
python train_tft.py --fold 0 1 2 4 6 7
```

## 참고 문헌

1. Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." International Journal of Forecasting.

2. Vaswani, A., et al. (2017). "Attention is All You Need." NIPS.

## 라이센스

MIT License

## 문의

이슈가 있으면 GitHub Issues에 등록해주세요.
