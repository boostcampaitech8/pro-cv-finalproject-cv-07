#export PYTHONPATH=/data/ephemeral/home/pro-cv-finalproject-cv-07/python
from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset import build_dataset, train_valid_split
from src.data.preprocessing import scale_train_data
from src.models.LSTM import LSTM
from src.engine.trainer import train
from src.utils.visualization import save_loss_curve

import os
import tyro
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


from src.data.dataset import deepar_split , build_multi_item_dataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger

def evaluate_model(predictor, dataset):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=100,
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

    print("\n========== Overall Metrics ==========")
    for k, v in agg_metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:20s}: {v:.4f}")


    return forecasts, tss, agg_metrics, item_metrics


# ======================================================
# 7. 미래 예측
# ======================================================
def predict_future(predictor, dataset, item_names):
    forecasts = list(predictor.predict(dataset))

    print("\n========== Future Forecast ==========")
    for name, forecast in zip(item_names, forecasts):
        median = forecast.median
        print(f"\n{name.upper()}")
        print(f"Last forecast value: {median[-1]:.2f}")
        print(
            f"90% CI: [{forecast.quantile(0.05)[-1]:.2f}, "
            f"{forecast.quantile(0.95)[-1]:.2f}]"
        )

    return forecasts


def train_deepar(
    dataset,
    prediction_length=30,
    context_length=60,
    num_feat_dynamic_real=0,
    epochs=30,
    batch_size=8,
    lr=1e-3,
):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    estimator = DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length,
        num_layers=3,
        hidden_size=64,
        dropout_rate=0.1,
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_parallel_samples=100,
        batch_size=32,
        trainer_kwargs={
            "max_epochs": epochs,
            "accelerator": accelerator,
            "devices": 1,
        }
    )
    

    predictor = estimator.train(dataset)
    return predictor
def plot_loss_from_logger(logger, save_path=None):
    csv_file = os.path.join(logger.log_dir, "metrics.csv")
    if not os.path.exists(csv_file):
        print(f"[WARN] {csv_file} 파일이 없습니다.")
        return

    df = pd.read_csv(csv_file)

    # 1. 핵심: Epoch별로 그룹화하여 NaN 값을 합칩니다.
    # 각 에폭에서 존재하는 값들만 뽑아서 평균(mean)을 내면 NaN이 사라집니다.
    metrics = df.groupby("epoch").agg({
        "train_loss": "mean",
        "val_loss": "mean"
    }).reset_index()

    # 2. 시각화
    plt.figure(figsize=(10, 5))
    
    # Train Loss (파란색)
    if not metrics["train_loss"].dropna().empty:
        plt.plot(metrics["epoch"], metrics["train_loss"], 
                 label='Train Loss', color='tab:blue', marker='o', markersize=4)
    
    # Val Loss (주황색)
    if not metrics["val_loss"].dropna().empty:
        plt.plot(metrics["epoch"], metrics["val_loss"], 
                 label='Val Loss', color='tab:orange', marker='s', markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (Negative Log-Likelihood)")
    plt.title("DeepAR Training & Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 3. 저장 및 출력
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"📈 Loss curve saved to: {save_path}")
    
    plt.show()
    plt.close()
    
    
def plot_predictions(forecasts, tss, item_ids, num_plots=3, save_dir=None):
    import matplotlib.pyplot as plt
    import pandas as pd

    for i, (forecast, ts, item_id) in enumerate(zip(forecasts, tss, item_ids)):
        if i >= num_plots:
            break

        fig, ax = plt.subplots(figsize=(10, 4))

        # ===== 실제값 =====
        ts_last = ts[-200:]

        if isinstance(ts_last.index, pd.PeriodIndex):
            ts_index = ts_last.index.to_timestamp()
        else:
            ts_index = ts_last.index

        ax.plot(
            ts_index,
            ts_last.values,
            label="actual"
        )

        # ===== 예측 =====
        forecast_index = pd.period_range(
            start=forecast.start_date,
            periods=len(forecast.mean),
            freq=forecast.freq,
        ).to_timestamp()

        ax.plot(
            forecast_index,
            forecast.mean,
            label="forecast_mean",
            color="tab:orange",
        )

        ax.fill_between(
            forecast_index,
            forecast.quantile(0.1),
            forecast.quantile(0.9),
            alpha=0.3,
            label="80% interval",
        )

        ax.set_title(f"Validation Prediction: {item_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 파일로 저장
        if save_dir:
            save_path = Path(save_dir) / f"prediction_{item_id}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.close()  # 메모리 해제


#아니 잠깐만 이거 날짜 어떻게 되있는거지? 나중에 그거 하면 해결될듯 . 전체 데이터 크기를 똑같이
def plot_loss_from_logger(logger, save_path=None):
    # CSVLogger는 metrics.csv 파일에 epoch별 loss 저장
    csv_file = os.path.join(logger.log_dir, "metrics.csv")
    df = pd.read_csv(csv_file)

    if 'train_loss' not in df.columns:
        print("[WARN] train_loss 컬럼이 없어서 plot 불가")
        return

    plt.figure(figsize=(8,4))
    plt.plot(df['train_loss'], label='train_loss')
    if 'val_loss' in df.columns:
        plt.plot(df['val_loss'], label='val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DeepAR Training Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def lag_features_by_1day(df: pd.DataFrame, feature_cols, group_col="item_id", time_col="time"):
    df = df.sort_values([group_col, time_col]).copy()

    # feature_cols만 1일 lag (누수 방지)
    df[feature_cols] = df.groupby(group_col)[feature_cols].shift(1)

    # lag로 생긴 NaN 제거(첫 날)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df

    
def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    dfs = {}
    for name in ["corn", "wheat", "soybean"]:
        data_path = os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv")
        data = pd.read_csv(data_path)
        data["item_id"] = name
        data['time']=pd.to_datetime(data['time'])
        dfs[name] = data
        print(len(data[data["item_id"]==name]))
    # feature_cols만 추출할 때 long_df 활용 가능
    feature_cols = [
        c for c in pd.concat(dfs.values(), ignore_index=True).columns
        if c not in ["time", "item_id","close"] and not c.startswith("log_return_")
    ]
    for name in list(dfs.keys()):
        dfs[name] = lag_features_by_1day(dfs[name], feature_cols, group_col="item_id", time_col="time")

    cfg.epochs=30
    cfg.fold=[0,1,2,3,4,5,6,7]
    for fold in cfg.fold:
        train_dfs= {}
        val_dfs = {}
        
        for name, df in dfs.items():
            print(1)
            train_df, val_df = deepar_split(
                df,
                os.path.join(cfg.data_dir, "rolling_fold.json"),
                fold,
            )
            train_dfs[name] = train_df
            val_dfs[name]   = val_df
        
        
        
        for h in cfg.horizons:
            print(2)
            train_ds = build_multi_item_dataset(
                train_dfs, f"log_return_{h}", feature_cols )
            print(f"horizon={h}, train_ds length={len(train_ds)}")
            val_ds = build_multi_item_dataset(
    val_dfs, f"log_return_{h}", feature_cols
)
            for entry in train_ds:
                print(entry["item_id"], len(entry["target"]))
            logger = CSVLogger(
        save_dir=cfg.checkpoint_dir,
        name=f"deepar_fold{fold}_h{h}"
    )
            estimator = DeepAREstimator(
                freq="D",
                prediction_length=h,
                context_length=cfg.seq_length,
                num_feat_dynamic_real=len(feature_cols),
                trainer_kwargs={
                    "max_epochs": cfg.epochs,
                    "accelerator": "gpu"
                    if torch.cuda.is_available()
                    else "cpu",
                    "devices": 1,"logger":logger,
                },)
            
                    # ------------------------------
            # 5️⃣ 모델 저장
            # ------------------------------
            predictor = estimator.train(
            training_data=train_ds,
            validation_data=val_ds,
        )

            # ===============================
            # 2️⃣ Validation 예측 생성
            # ===============================

            forecast_it, ts_it = make_evaluation_predictions(
                dataset=val_ds,
                predictor=predictor,
                num_samples=100,
            )

            forecasts = list(forecast_it)
            tss       = list(ts_it)

            # ===============================
            # 3️⃣ Validation Metric 계산
            # ===============================
            evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
            agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

            print("\n=== Validation Metrics ===")
            for k, v in agg_metrics.items():
                if isinstance(v, (int, float)):
                    print(f"{k:20s}: {v:.4f}")

            # ===============================
            # 4️⃣ 모델 저장
            # ===============================
            save_path = Path(cfg.checkpoint_dir) / f"deepar_multi_fold{fold}_h{h}"
            save_path.mkdir(parents=True, exist_ok=True)
            predictor.serialize(save_path)

            # ===============================
            # 5️⃣ Loss curve 시각화
            # ===============================
            plot_loss_from_logger(
                logger,
                save_path=save_path / "loss_curve.png",
            )

            plot_predictions(
    forecasts=forecasts,
    tss=tss,
    item_ids=list(val_dfs.keys()),  # corn / wheat / soybean
    num_plots=3,
    save_dir=save_path,  # 👈 저장 경로 추가
)

            print(f"\n✅ Plots saved to: {save_path}")

                                

           
if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)
    
    