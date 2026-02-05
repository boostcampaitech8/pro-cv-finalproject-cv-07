
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset import build_multi_item_dataset, deepar_split, lag_features_by_1day
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


# ===============================
# 📐 Log Return → Close 복원
# ===============================
def log_return_to_close(log_returns, initial_close):
    """
    log_return[t] = log(close[t] / close[t-1])
    → close[t] = close[t-1] * exp(log_return[t])
    
    Args:
        log_returns: numpy array of log returns (length T)
        initial_close: close price at t=-1 (직전 값)
    
    Returns:
        close_prices: numpy array (length T)
    """
    close_prices = np.zeros_like(log_returns)
    close_prices[0] = initial_close * np.exp(log_returns[0])
    
    for i in range(1, len(log_returns)):
        close_prices[i] = close_prices[i-1] * np.exp(log_returns[i])
    
    return close_prices


def get_initial_close(df, item_id, split_date):
    """
    validation 시작 직전의 close 값을 가져옴
    
    Args:
        df: 전체 DataFrame (item_id, time, close 포함)
        item_id: 'corn', 'wheat', 'soybean' 등
        split_date: validation 시작 날짜
    
    Returns:
        float: 직전 close 가격
    """
    item_df = df[df['item_id'] == item_id].copy()
    item_df = item_df.sort_values('time')
    
    # validation 시작 직전 행
    prev_df = item_df[item_df['time'] < split_date]
    
    if len(prev_df) == 0:
        raise ValueError(f"No data before {split_date} for {item_id}")
    
    return prev_df.iloc[-1]['close']


# ===============================
# 📊 Close 기반 Metric 계산
# ===============================
def compute_close_metrics(pred_close, true_close):
    """
    실제 가격 기준 MAE, RMSE, MAPE 계산
    """
    mae = np.mean(np.abs(pred_close - true_close))
    rmse = np.sqrt(np.mean((pred_close - true_close)**2))
    mape = np.mean(np.abs((pred_close - true_close) / true_close)) * 100
    
    # Directional Accuracy (가격 방향)
    # 현재 가격 대비 다음 가격이 올랐는지/내렸는지
    pred_direction = np.sign(np.diff(pred_close, prepend=pred_close[0]))
    true_direction = np.sign(np.diff(true_close, prepend=true_close[0]))
    
    direction_acc = (pred_direction == true_direction).mean()
    
    return {
        "MAE_close": float(mae),
        "RMSE_close": float(rmse),
        "MAPE_close": float(mape),
        "Direction_Accuracy_close": float(direction_acc)
    }


# ===============================
# 📈 Close 시각화
# ===============================
def plot_close_predictions(
    forecasts, 
    tss, 
    item_ids, 
    initial_closes,
    save_dir=None,
    num_plots=3
):
    """
    Log Return 예측 → Close로 복원 → 시각화
    
    Args:
        forecasts: GluonTS Forecast 객체 리스트
        tss: 실제 log return 시계열
        item_ids: ['corn', 'wheat', 'soybean']
        initial_closes: dict {item_id: 직전 close 값}
        save_dir: 저장 경로
    """
    for i, (forecast, ts, item_id) in enumerate(zip(forecasts, tss, item_ids)):
        if i >= num_plots:
            break
        
        # ===== 1️⃣ Log Return → Close 복원 =====
        # 실제값
        true_log_returns = ts.values[-len(forecast.mean):]
        true_close = log_return_to_close(true_log_returns, initial_closes[item_id])
        
        # 예측값 (mean)
        pred_log_returns = forecast.mean
        pred_close_mean = log_return_to_close(pred_log_returns, initial_closes[item_id])
        
        # 예측값 (0.1, 0.9 quantile)
        pred_log_q10 = forecast.quantile(0.1)
        pred_log_q90 = forecast.quantile(0.9)
        
        pred_close_q10 = log_return_to_close(pred_log_q10, initial_closes[item_id])
        pred_close_q90 = log_return_to_close(pred_log_q90, initial_closes[item_id])
        
        # ===== 2️⃣ 시각화 =====
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # 날짜 인덱스
        forecast_index = pd.period_range(
            start=forecast.start_date,
            periods=len(forecast.mean),
            freq=forecast.freq,
        ).to_timestamp()
        
        # 실제 Close
        ax.plot(
            forecast_index,
            true_close,
            label="Actual Close",
            color="black",
            linewidth=2,
            marker='o',
            markersize=4
        )
        
        # 예측 Close (Mean)
        ax.plot(
            forecast_index,
            pred_close_mean,
            label="Predicted Close (Mean)",
            color="tab:orange",
            linewidth=2,
            linestyle='--',
            marker='s',
            markersize=4
        )
        
        # 80% Confidence Interval
        ax.fill_between(
            forecast_index,
            pred_close_q10,
            pred_close_q90,
            alpha=0.3,
            color="tab:orange",
            label="80% Prediction Interval"
        )
        
        # ===== 3️⃣ Metric 출력 =====
        metrics = compute_close_metrics(pred_close_mean, true_close)
        
        textstr = f"MAE: ${metrics['MAE_close']:.2f}\n"
        textstr += f"RMSE: ${metrics['RMSE_close']:.2f}\n"
        textstr += f"MAPE: {metrics['MAPE_close']:.2f}%\n"
        textstr += f"Direction Acc: {metrics['Direction_Accuracy_close']:.2%}"
        
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_title(f"Close Price Prediction: {item_id.upper()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Close Price ($)", fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # ===== 4️⃣ 저장 =====
        if save_dir:
            save_path = Path(save_dir) / f"close_prediction_{item_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved close plot to: {save_path}")
        
        plt.close()


# ===============================
# 📋 Close Metric 집계
# ===============================
def collect_close_metrics(forecasts, tss, item_ids, initial_closes, fold, horizon):
    """
    모든 item의 Close 기반 metric을 수집
    """
    results = []
    
    for forecast, ts, item_id in zip(forecasts, tss, item_ids):
        # Log Return → Close 복원
        true_log_returns = ts.values[-len(forecast.mean):]
        true_close = log_return_to_close(true_log_returns, initial_closes[item_id])
        
        pred_log_returns = forecast.mean
        pred_close_mean = log_return_to_close(pred_log_returns, initial_closes[item_id])
        
        # Metric 계산
        metrics = compute_close_metrics(pred_close_mean, true_close)
        metrics.update({
            "item": item_id,
            "fold": fold,
            "horizon": horizon
        })
        
        results.append(metrics)
    
    return results


# ===============================
# 기존 Log Return Metric 함수들
# ===============================
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


def compute_metrics(forecasts, tss, fold, horizon):
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
    
    metrics = {
        "MAE": compute_mae(forecasts, tss),
        "RMSE": agg_metrics.get("RMSE"),
        "MAPE": agg_metrics.get("MAPE"),
        "Directional_Accuracy": directional_accuracy(forecasts, tss),
        "fold": fold,
        "horizon": horizon
    }
    
    return metrics


def plot_loss_from_logger(logger, save_path=None):
    csv_file = os.path.join(logger.log_dir, "metrics.csv")
    if not os.path.exists(csv_file):
        print(f"[WARN] {csv_file} 파일이 없습니다.")
        return

    df = pd.read_csv(csv_file)

    if "epoch" in df.columns:
        cols = [c for c in ["train_loss", "val_loss"] if c in df.columns]
        if not cols:
            print("[WARN] train_loss / val_loss 컬럼이 없습니다.")
            return

        metrics = df.groupby("epoch")[cols].mean().reset_index()
        x = metrics["epoch"]
        xlabel = "Epoch"

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


# ===============================
# 🚀 Main 함수
# ===============================
def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # ===============================
    # 1️⃣ 데이터 로드
    # ===============================
    print("\n" + "="*60)
    print("📁 Loading Dataset")
    print("="*60)
    
    dfs = {}
    for name in ["corn", "wheat", "soybean", "gold", "silver", "copper"]:
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

    cfg.epochs = 30
    cfg.fold = [7]

    all_results_log = []  # Log Return 기반 metric
    all_results_close = []  # Close 기반 metric

    # ===============================
    # 3️⃣ Fold별 학습
    # ===============================
    for fold in cfg.fold:
        train_dfs = {}
        val_dfs = {}
        
        for name, df in dfs.items():
            train_df, val_df = deepar_split(
                df,
                os.path.join(cfg.data_dir, "rolling_fold.json"),
                fold,
            )
            train_dfs[name] = train_df
            val_dfs[name] = val_df
        
        # ===== Validation 시작 날짜 추출 =====
        val_start_dates = {}
        for name, val_df in val_dfs.items():
            val_start_dates[name] = val_df['time'].min()
        
        # ===== 직전 Close 값 추출 =====
        initial_closes = {}
        for name in dfs.keys():
            initial_closes[name] = get_initial_close(
                dfs[name], name, val_start_dates[name]
            )
            print(f"📌 {name} initial close: ${initial_closes[name]:.2f}")
        
        # ===============================
        # 4️⃣ Horizon별 학습
        # ===============================
        for h in cfg.horizons:
            print("\n" + "="*60)
            print(f"🚀 Fold {fold} | Horizon {h}")
            print("="*60)
            
            train_ds = build_multi_item_dataset(train_dfs, f"log_return_{h}", feature_cols)
            val_ds = build_multi_item_dataset(val_dfs, f"log_return_{h}", feature_cols)
            
            logger = CSVLogger(
                save_dir=cfg.checkpoint_dir,
                name=f"deepar_fold{fold}_h{h}"
            )
            
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
            
            predictor = estimator.train(
                training_data=train_ds,
                validation_data=val_ds,
            )
            
            # ===============================
            # 5️⃣ Validation 예측 생성
            # ===============================
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=val_ds,
                predictor=predictor,
                num_samples=100,
            )
            
            forecasts = list(forecast_it)
            tss = list(ts_it)
            
            # ===============================
            # 6️⃣ Log Return 기반 Metric
            # ===============================
            evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
            agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
            metrics = compute_metrics(forecasts, tss, fold, h)
            
            all_results_log.append({
                "item": "ALL",
                "fold": fold,
                "horizon": h,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "Directional_Accuracy": metrics["Directional_Accuracy"],
            })
            
            item_ids = list(val_dfs.keys())
            
            for item_id, ts, fcst in zip(item_ids, tss, forecasts):
                metrics = compute_metrics([fcst], [ts], fold, h)
                all_results_log.append({
                    "item": item_id,
                    "fold": fold,
                    "horizon": h,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics["MAPE"],
                    "Directional_Accuracy": metrics["Directional_Accuracy"],
                })
            
            # ===============================
            # 7️⃣ Close 기반 Metric
            # ===============================
            close_metrics = collect_close_metrics(
                forecasts, tss, item_ids, initial_closes, fold, h
            )
            all_results_close.extend(close_metrics)
            
            # ===============================
            # 8️⃣ 시각화 저장
            # ===============================
            save_path = Path(cfg.checkpoint_dir) / f"deepar_multi_fold{fold}_h{h}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            predictor.serialize(save_path)
            
            plot_loss_from_logger(logger, save_path=save_path / "loss_curve.png")
            
            # Close 기반 시각화
            plot_close_predictions(
                forecasts=forecasts,
                tss=tss,
                item_ids=item_ids,
                initial_closes=initial_closes,
                save_dir=save_path,
                num_plots=len(item_ids)
            )
            
            print(f"\n✅ Saved to: {save_path}")

    # ===============================
    # 9️⃣ 최종 결과 저장
    # ===============================
    # Log Return Metric
    log_df = pd.DataFrame(all_results_log)
    log_csv = Path(cfg.checkpoint_dir) / "deepar_metrics_log_return.csv"
    log_df.to_csv(log_csv, index=False, float_format="%.6f")
    print(f"\n✅ Log Return Metrics saved to: {log_csv}")
    
    # Close Metric
    close_df = pd.DataFrame(all_results_close)
    close_csv = Path(cfg.checkpoint_dir) / "deepar_metrics_close.csv"
    close_df.to_csv(close_csv, index=False, float_format="%.6f")
    print(f"✅ Close Metrics saved to: {close_csv}")
    
    print("\n" + "="*60)
    print("📊 Log Return Metrics Summary")
    print("="*60)
    print(log_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("💵 Close Price Metrics Summary")
    print("="*60)
    print(close_df.to_string(index=False))


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)
