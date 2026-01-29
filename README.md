# 주의

- 모든 명령어는 python 디렉토리 내부에서 실행
- 환경변수 설정 필수
```Bash
export PYTHONPATH=$PYTHONPATH:.
```
- 파일 구조 변경 ⭐⭐⭐
```Bash
.
|-- README.md
`-- python
    |-- etc
    |   `-- EDA.ipynb
    |-- predictVLM
    |   |-- scripts
    |   `-- src
    |-- shared
    |   `-- datasets
    |       |-- corn_future_price.csv
    |       |-- news_articles_resources.csv
    |       |-- news_articles_resources_entities.csv
    |       |-- resource_article_clustering.html
    |       |-- rolling_fold.json
    |       |-- soybean_future_price.csv
    |       `-- wheat_future_price.csv
    `-- transformer
        |-- scripts
        |   |-- preprocessing.py
        |   |-- test.py
        |   `-- train.py
        `-- src
            |-- configs
            |   `-- train_config.py
            |-- data
            |   |-- dataset.py
            |   |-- feature_engineering.py
            |   |-- postprocessing.py
            |   `-- preprocessing.py
            |-- engine
            |   |-- inference.py
            |   `-- trainer.py
            |-- metrics
            |   `-- metrics.py
            |-- models
            |   `-- LSTM.py
            |-- outputs
            `-- utils
                |-- set_seed.py
                `-- visualization.py
```
- 제일 중요한거 그냥 제 나름대로 code refactoring한거라 이상한거 많을수도 있음 코드 돌리면서 자유롭게 수정하셈

# 실행

## 데이터 전처리

- 최초 실행 시, 오래 걸릴 수 있음
- 빠르게 실행하고 싶다면 노션에 전처리한 csv 파일 올려뒀으니 다운로드 후, python/src/datasets/preprocessing 폴더 생성 후 내부에 저장
```Bash
python scripts/preprocessing.py --use_ema_features --use_volatility_features --use_news_count_features --use_news_imformation_features
```

## 학습

- 기본 코드는 t-20 ~ t-1 시점을 이용해서 t, t+4, t+9, t+19를 예측하도록 설정되어있음
- epochs = 300 이 외의 기본 값은 train_config.py에서 확인
- 만약, CLI로 파라미터 수정하면 test 실행할 때도 반영해서 실행하셈!!
- 앙상블 코드도 구현은 해두었으나 성능이 그닥,,, 최근 데이터에 가중치 주는 방법으로 해야 성능이 좋을듯
```Bash
python scripts/train.py
```

## 예측

```Bash
python scripts/test.py
```

# Baseline 실험 결과

## Corn
### t 시점 예측 (horizon = 1)
```Bash
Horizon 1 Metrics:
  RMSE: 0.009050
  MAE: 0.006892
  MAPE: 593103.25%
  R²: 0.5104
  DA: 77.16%
```

### t+4 시점 예측 (horizon = 5)
```Bash
Horizon 5 Metrics:
  RMSE: 0.009998
  MAE: 0.007714
  MAPE: 185047.97%
  R²: 0.8656
  DA: 87.07%
```

### t+9 시점 예측 (horizon = 10)
```Bash
Horizon 10 Metrics:
  RMSE: 0.010853
  MAE: 0.008314
  MAPE: 1354671.12%
  R²: 0.9227
  DA: 90.52%
```

### t+19 시점 예측 (horizon = 20)
```Bash
Horizon 20 Metrics:
  RMSE: 0.015474
  MAE: 0.011589
  MAPE: 713698.06%
  R²: 0.9290
  DA: 95.26%
```

## Wheat

### t 시점 예측 (horizon = 1)
```Bash
Horizon 1 Metrics:
  RMSE: 0.002705
  MAE: 0.002106
  MAPE: 195197.27%
  R²: 0.9639
  DA: 95.69%
```

### t+4 시점 예측 (horizon = 5)
```Bash
Horizon 5 Metrics:
  RMSE: 0.009671
  MAE: 0.007656
  MAPE: 776731.56%
  R²: 0.9002
  DA: 90.52%
```

### t+9 시점 예측 (horizon = 10)
```Bash
Horizon 10 Metrics:
  RMSE: 0.012551
  MAE: 0.010115
  MAPE: 350078.81%
  R²: 0.8870
  DA: 88.79%
```

### t+19 시점 예측 (horizon = 20)
```Bash
Horizon 20 Metrics:
  RMSE: 0.017791
  MAE: 0.013580
  MAPE: 1879892.00%
  R²: 0.8500
  DA: 90.09%
```

## Soybean

### t 시점 예측 (horizon = 1)
```Bash
Horizon 1 Metrics:
  RMSE: 0.002698
  MAE: 0.002128
  MAPE: 252499.12%
  R²: 0.9523
  DA: 93.97%
```

### t+4 시점 예측 (horizon = 5)
```Bash
Horizon 5 Metrics:
  RMSE: 0.006612
  MAE: 0.005073
  MAPE: 275038.19%
  R²: 0.9353
  DA: 90.52%
```

### t+9 시점 예측 (horizon = 10)
```Bash
Horizon 10 Metrics:
  RMSE: 0.009352
  MAE: 0.007123
  MAPE: 484500.72%
  R²: 0.9344
  DA: 87.07%
```

### t+19 시점 예측 (horizon = 20)
```Bash
Horizon 20 Metrics:
  RMSE: 0.015300
  MAE: 0.012341
  MAPE: 716374.81%
  R²: 0.9063
  DA: 90.52%
```
