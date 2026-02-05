# train_deepar_full.py
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset import build_multi_item_dataset, lag_features_by_1day
from collections import defaultdict
import tyro
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from pytorch_lightning.loggers import CSVLogger


def compute_mae(forecasts, tss):
    maes = []
    for fcst, ts in zip(forecasts, tss):
        y_true = ts.values[-len(fcst.mean):]
        y_pred = fcst.mean
        maes.append(np.mean(np.abs(y_true - y_pred)))
    return float(np.mean(maes))


def directional_accuracy(forecasts, tss):
    accs = []
    for forecast, ts in zip(forecasts, tss):
        y_true = ts.values[-len(forecast.mean):]
        y_pred = forecast.mean
        sign_true = np.sign(y_true)
        sign_pred = np.sign(y_pred)
        acc = (sign_true == sign_pred).mean()
        accs.append(acc)
    return float(np.mean(accs))


def compute_metrics(forecasts, tss, horizon):
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
    
    metrics = {
        "MAE": compute_mae(forecasts, tss),
        "RMSE": agg_metrics.get("RMSE"),
        "MAPE": agg_metrics.get("MAPE"),
        "Directional_Accuracy": directional_accuracy(forecasts, tss),
        "horizon": horizon
    }
    
    return metrics, agg_metrics, item_metrics


def plot_predictions(forecasts, tss, item_ids, num_plots=3, save_dir=None):
    import matplotlib.pyplot as plt
    import pandas as pd

    for i, (forecast, ts, item_id) in enumerate(zip(forecasts, tss, item_ids)):
        if i >= num_plots:
            break

        fig, ax = plt.subplots(figsize=(10, 4))

        # 실제값
        ts_last = ts[-200:]

        if isinstance(ts_last.index, pd.PeriodIndex):
            ts_index = ts_last.index.to_timestamp()
        else:
            ts_index = ts_last.index

        ax.plot(ts_index, ts_last.values, label="actual")

        # 예측
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

        ax.set_title(f"Prediction: {item_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"prediction_{item_id}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"📊 Saved plot to: {save_path}")
        
        plt.close()


def plot_loss_from_logger(logger, save_path=None):
    csv_file = os.path.join(logger.log_dir, "metrics.csv")
    if not os.path.exists(csv_file):
        print(f"[WARN] {csv_file} 파일이 없습니다.")
        return

    df = pd.read_csv(csv_file)

    # epoch 컬럼이 있는 경우
    if "epoch" in df.columns:
        cols = [c for c in ["train_loss", "val_loss"] if c in df.columns]
        if not cols:
            print("[WARN] train_loss / val_loss 컬럼이 없습니다.")
            return

        metrics = df.groupby("epoch")[cols].mean().reset_index()
        x = metrics["epoch"]
        xlabel = "Epoch"

    # step만 있는 경우
    elif "step" in df.columns:
        cols = [c for c in ["train_loss", "val_loss"] if c in df.columns]
        if not cols:
            print("[WARN] train_loss / val_loss 컬럼이 없습니다.")
            return

        metrics = df[["step"] + cols].copy()
        x = metrics["step"]
        xlabel = "Training Step"

    else:
        print("[WARN] epoch/step 컬럼이 없어 loss curve를 그릴 수 없습니다.")
        return

    # 시각화
    plt.figure(figsize=(10, 5))

    if "train_loss" in metrics and metrics["train_loss"].dropna().any():
        plt.plot(x, metrics["train_loss"], label="Train Loss", marker="o", markersize=3)

    if "val_loss" in metrics and metrics["val_loss"].dropna().any():
        plt.plot(x, metrics["val_loss"], label="Validation Loss", marker="s", markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel("Loss (Negative Log-Likelihood)")
    plt.title("DeepAR Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 Loss curve saved to: {save_path}")

    plt.close()


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # ===============================
    # 1️⃣ 전체 데이터 로드
    # ===============================
    print("\n" + "="*60)
    print("📁 Loading Full Dataset (No Fold Split)")
    print("="*60)
    
    dfs = {}
    for name in ["corn", "wheat", "soybean"]:
        data_path = os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv")
        data = pd.read_csv(data_path)
        data["item_id"] = name
        data['time'] = pd.to_datetime(data['time'])
        dfs[name] = data
        print(f"✅ {name}: {len(data)} samples")

    # ===============================
    # 2️⃣ Feature 추출 + Lag
    # ===============================
    feature_cols = [
        c for c in pd.concat(dfs.values(), ignore_index=True).columns
        if c not in ["time", "item_id", "close"] and not c.startswith("log_return_")
    ]
    
    print(f"\n🔧 Feature columns: {len(feature_cols)}")
    
    for name in list(dfs.keys()):
        dfs[name] = lag_features_by_1day(
            dfs[name], feature_cols, group_col="item_id", time_col="time"
        )
        print(f"   {name} after lag: {len(dfs[name])} samples")

    # ===============================
    # 3️⃣ Horizon별 학습
    # ===============================
    cfg.epochs = 30
    all_results = []

    for h in cfg.horizons:
        print("\n" + "="*60)
        print(f"🚀 Training Horizon = {h}")
        print("="*60)
        
        # ===== 데이터셋 생성 (전체 데이터) =====
        full_ds = build_multi_item_dataset(
            dfs, f"log_return_{h}", feature_cols
        )
        print(f"📦 Dataset size: {len(full_ds)} items")
        
        for entry in full_ds:
            print(f"   {entry['item_id']}: {len(entry['target'])} timesteps")

        # ===== Logger =====
        logger = CSVLogger(
            save_dir=cfg.checkpoint_dir,
            name=f"deepar_full_h{h}"
        )

        # ===== Estimator =====
        estimator = DeepAREstimator(
            freq="D",
            prediction_length=h,
            context_length=cfg.seq_length,
            num_feat_dynamic_real=len(feature_cols),
            num_layers=3,
            hidden_size=64,
            dropout_rate=0.1,
            trainer_kwargs={
                "max_epochs": cfg.epochs,
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
                "logger": logger,
            },
        )

        # ===== 학습 =====
        print(f"\n⏳ Training for {cfg.epochs} epochs...")
        predictor = estimator.train(training_data=full_ds)

        # ===============================
        # 4️⃣ 전체 데이터로 예측 (In-sample)
        # ===============================
        print("\n📊 Generating predictions on full dataset...")
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=full_ds,
            predictor=predictor,
            num_samples=100,
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        # ===============================
        # 5️⃣ Metric 계산
        # ===============================
        metrics, agg_metrics, item_metrics = compute_metrics(forecasts, tss, h)

        # Aggregate 저장
        all_results.append({
            "item": "ALL",
            "horizon": h,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"],
            "Directional_Accuracy": metrics["Directional_Accuracy"],
        })

        # Item별 저장
        item_metrics_df = item_metrics.reset_index()
        item_metrics_df.rename(columns={"index": "item_id"}, inplace=True, errors="ignore")
        item_metrics_df["horizon"] = h
        
        # Directional Accuracy 추가
        item_ids = list(dfs.keys())
        for item_id, ts, fcst in zip(item_ids, tss, forecasts):
            da = directional_accuracy([fcst], [ts])
            item_metrics_df.loc[
                item_metrics_df.get("item_id", item_metrics_df.index) == item_id, 
                "Directional_Accuracy"
            ] = da

        all_results.extend(item_metrics_df.to_dict("records"))

        # 출력
        print("\n" + "="*60)
        print("📈 Overall Metrics")
        print("="*60)
        for k, v in agg_metrics.items():
            if isinstance(v, (int, float)):
                print(f"{k:25s}: {v:.6f}")

        print("\n📊 Item-wise Metrics")
        print(item_metrics_df.to_string(index=False))

        # ===============================
        # 6️⃣ 모델 저장
        # ===============================
        save_path = Path(cfg.checkpoint_dir) / f"deepar_full_h{h}"
        save_path.mkdir(parents=True, exist_ok=True)
        predictor.serialize(save_path)
        print(f"\n💾 Model saved to: {save_path}")

        # ===============================
        # 7️⃣ 시각화
        # ===============================
        plot_loss_from_logger(logger, save_path=save_path / "loss_curve.png")
        
        plot_predictions(
            forecasts=forecasts,
            tss=tss,
            item_ids=item_ids,
            num_plots=len(item_ids),
            save_dir=save_path,
        )

    # ===============================
    # 8️⃣ 최종 결과저장
    # ===============================
    results_df = pd.DataFrame(all_results)
    save_csv = Path(cfg.checkpoint_dir) / "deepar_full_metrics_summary.csv"
    results_df.to_csv(save_csv, index=False, float_format="%.6f")

    print("\n" + "="*60)
    print(f"✅ All metrics saved to: {save_csv}")
    print("="*60)
    print(results_df)

    return all_results


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    all_results = main(cfg)
