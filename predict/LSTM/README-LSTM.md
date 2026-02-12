# 실행

## 데이터 전처리

- 최초 실행 시, 기사 데이터 전처리 문제로 오래 걸릴 수 있음
```Bash
python predict/LSTM/scripts/preprocessing.py --use_ema_features --use_volatility_features --use_news_count_features --use_news_imformation_features
```

## 학습

- 기본 코드는 t-20 ~ t-1 시점을 이용해서 t, t+4, t+9, t+19를 예측하도록 설정되어있음
- epochs = 300 이 외의 기본 값은 train_config.py에서 확인
- fold 별 앙상블 가능하게 구현하긴 했으나, 성능이 좋진 않음
```Bash
python predict/LSTM/scripts/train.py
```

## 예측

```Bash
python predict/LSTM/scripts/test.py
```