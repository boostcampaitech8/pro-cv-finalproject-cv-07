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
import json

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from pytorch_lightning.loggers import CSVLogger


# ===============================
# 🛡️ 데이터 검증 함수
# ===============================
def validate_dataset(dataset, dataset_name="dataset"):
    """
    Dataset의 NaN, Inf 값 체크
    """
    print(f"\n🔍 Validating {dataset_name}...")
    
    issues_found = False
    
    for i, entry in enumerate(dataset):
        item_id = entry.get('item_id', f'item_{i}')
        target = entry['target']
        
        # Target 검증
        target_array = np.array(target)
        
        if np.any(np.isnan(target_array)):
            print(f"  ❌ {item_id}: Target contains NaN!")
            print(f"     NaN count: {np.sum(np.isnan(target_array))}")
            issues_found = True
        
        if np.any(np.isinf(target_array)):
            print(f"  ❌ {item_id}: Target contains Inf!")
            print(f"     Inf count: {np.sum(np.isinf(target_array))}")
            issues_found = True
        
        # Feature 검증
        if 'feat_dynamic_real' in entry:
            feat = entry['feat_dynamic_real']
            feat_array = np.array(feat)
            
            if np.any(np.isnan(feat_array)):
                nan_count = np.sum(np.isnan(feat_array))
                print(f"  ❌ {item_id}: Features contain {nan_count} NaN values!")
                issues_found = True
            
            if np.any(np.isinf(feat_array)):
                inf_count = np.sum(np.isinf(feat_array))
                print(f"  ❌ {item_id}: Features contain {inf_count} Inf values!")
                issues_found = True
        
        # Target 통계
        if i < 3:  # 처음 3개만 출력
            print(f"  📊 {item_id}: target range = [{target_array.min():.6f}, {target_array.max():.6f}], "
                  f"mean = {target_array.mean():.6f}, std = {target_array.std():.6f}")
    
    if not issues_found:
        print(f"  ✅ All data validated successfully!")
    else:
        print(f"  ⚠️ Issues found! Please fix data before training.")
        raise ValueError(f"{dataset_name} contains invalid values!")
    
    return True


def clean_features(df, feature_cols):
    """
    Feature의 NaN, Inf 값 처리
    """
    print(f"\n🧹 Cleaning features for item: {df['item_id'].iloc[0] if 'item_id' in df.columns else 'unknown'}")
    
    initial_nan_count = df[feature_cols].isna().sum().sum()
    initial_inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    
    print(f"  Initial: {initial_nan_count} NaN, {initial_inf_count} Inf values")
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # NaN 처리
        if df[col].isna().any():
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Inf 처리
        if np.isinf(df[col]).any():
            median_val = df[col].replace([np.inf, -np.inf], np.nan).median()
            if np.isnan(median_val):
                median_val = 0.0
            df[col] = df[col].replace([np.inf, -np.inf], median_val)
    
    final_nan_count = df[feature_cols].isna().sum().sum()
    final_inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    
    print(f"  Final: {final_nan_count} NaN, {final_inf_count} Inf values")
    
    return df


# ===============================
# 📐 Log Return → Close 복원
# ===============================
def log_return_to_close(log_returns, initial_close):
    """
    log_return[t] = log(close[t] / close[t-1])
    → close[t] = close[t-1] * exp(log_return[t])
    """
    close_prices = np.zeros_like(log_returns)
    close_prices[0] = initial_close * np.exp(log_returns[0])
    
    for i in range(1, len(log_returns)):
        close_prices[i] = close_prices[i-1] * np.exp(log_returns[i])
    
    return close_prices


def get_initial_close(original_df, item_id, val_start_date_str):
    """
    validation 시작 직전의 close 값을 가져옴
    """
    item_df = original_df[original_df['item_id'] == item_id].copy()
    item_df = item_df.sort_values('time').reset_index(drop=True)
    
    item_df['date_str'] = item_df['time'].astype(str).str[:10]
    prev_df = item_df[item_df['date_str'] < val_start_date_str]
    
    if prev_df.empty:
        raise ValueError(
            f"{item_id}: no data before validation start {val_start_date_str}\n"
            f"Available date range: {item_df['date_str'].min()} ~ {item_df['date_str'].max()}"
        )
    
    initial_close = prev_df.iloc[-1]['close']
    initial_date = prev_df.iloc[-1]['date_str']
    
    print(f"  ✅ {item_id}: initial_close = ${initial_close:.2f} (date: {initial_date})")
    
    return initial_close


# ===============================
# 📊 Close 기반 Metric 계산
# ===============================
def compute_close_metrics(pred_close, true_close):
    """
    실제 가격 기준 MAE, RMSE, MAPE 계산
    """
    true_close = np.asarray(true_close).reshape(-1)
    pred_close = np.asarray(pred_close).reshape(-1)
    
    if len(true_close) == 0 or len(pred_close) == 0:
        return {
            "MAE_close": 0.0,
            "RMSE_close": 0.0,
            "MAPE_close": 0.0,
            "Direction_Accuracy_close": 0.0
        }
    
    mae = np.mean(np.abs(pred_close - true_close))
    rmse = np.sqrt(np.mean((pred_close - true_close)**2))
    mape = np.mean(np.abs((pred_close - true_close) / (true_close + 1e-8))) * 100
    
    # Directional Accuracy (가격 방향)
    if len(pred_close) > 1:
        pred_direction = np.sign(np.diff(pred_close))
        true_direction = np.sign(np.diff(true_close))
        direction_acc = (pred_direction == true_direction).mean()
    else:
        direction_acc = 0.0
    
    return {
        "MAE_close": float(mae),
        "RMSE_close": float(rmse),
        "MAPE_close": float(mape),
        "Direction_Accuracy_close": float(direction_acc)
    }

import numpy as np
import pandas as pd
import traceback
from gluonts.dataset.common import ListDataset

def rolling_predictions(
    predictor,
    val_ds,
    horizon: int,
    context_length: int,
    num_samples: int = 100,
    max_steps: int | None = None,
):
    """
    Rolling 1-step-ahead evaluation.

    predictor: gluonts Predictor (DeepAR predictor)
    val_ds: ListDataset (validation dataset with 'target', optional 'feat_dynamic_real')
    horizon: predictor.prediction_length (e.g., 1/5/20)  # 개념상 동일
    context_length: predictor.context_length (e.g., cfg.seq_length)
    num_samples: sampling for probabilistic forecast
    max_steps: if not None, limit number of rolling steps per item (debug)
    """
    val_list = list(val_ds)
    results = []

    print(f"\n🔄 Rolling Predictions for {len(val_list)} items")
    print(f"   context_length={context_length}, horizon(prediction_length)={horizon}")

    for entry in val_list:
        item_id = entry.get("item_id", "unknown")
        target = np.asarray(entry["target"], dtype=np.float32)
        feat = entry.get("feat_dynamic_real", None)

        T = len(target)
        if feat is not None:
            feat = np.asarray(feat, dtype=np.float32)
            assert feat.ndim == 2, f"{item_id}: feat_dynamic_real must be 2D, got {feat.shape}"
            assert feat.shape[1] == T, f"{item_id}: feat length mismatch feat={feat.shape}, target={target.shape}"

        print(f"  📊 {item_id}: Total length={T}, feat={'yes' if feat is not None else 'no'}")

        # rolling 시작: 최소 context_length 확보되어야 함
        start_t = context_length
        if start_t >= T:
            print(f"    ⚠️ {item_id}: not enough points (need >={context_length}, have {T})")
            results.append({"item_id": item_id, "predictions": np.array([]), "actuals": np.array([])})
            continue

        preds, acts = [], []

        # 최대 스텝 제한(디버그)
        end_t = T
        if max_steps is not None:
            end_t = min(end_t, start_t + max_steps)

        for t in range(start_t, end_t):
            context_start = t - context_length
            y_ctx = target[context_start:t]                  # (context_length,)
            x_ctx = feat[:, context_start:t] if feat is not None else None  # (F, context_length)

            # ✅ start를 window 시작 시점으로 이동
            temp_entry = {
                "start": entry["start"] + pd.Timedelta(days=context_start),
                "target": y_ctx,
            }
            if x_ctx is not None:
                temp_entry["feat_dynamic_real"] = x_ctx
            if "item_id" in entry:
                temp_entry["item_id"] = entry["item_id"]

            # freq는 val_ds가 가진 freq를 그대로(문자열이면 그대로 사용 가능)
            temp_ds = ListDataset([temp_entry], freq=entry.get("freq", "D"))

            try:
                forecast = next(iter(predictor.predict(temp_ds, num_samples=num_samples)))

                # 1-step-ahead만 평가 (horizon이 5여도 첫 값만)
                pred_1 = float(forecast.mean[0])
                act_1 = float(target[t])

                preds.append(pred_1)
                acts.append(act_1)

            except Exception as e:
                print(f"    ❌ {item_id} error at t={t}/{T}: {e}")
                print(f"       y_ctx shape: {y_ctx.shape}")
                if x_ctx is not None:
                    print(f"       x_ctx shape: {x_ctx.shape}")
                traceback.print_exc()
                break

        print(f"  ✅ {item_id}: Collected {len(preds)} predictions")

        results.append(
            {"item_id": item_id, "predictions": np.asarray(preds), "actuals": np.asarray(acts)}
        )

    return results



# ===============================
# 📈 시각화 함수
# ===============================
def plot_close_predictions_full(
    rolling_results,
    initial_closes,
    val_start_date_str,
    horizon,
    save_dir=None,
    num_plots=6
):
    """
    Validation 전체 기간의 Close 예측 시각화
    """
    for i, result in enumerate(rolling_results):
        if i >= num_plots:
            break
        
        item_id = result['item_id']
        pred_log_returns = result['predictions']
        true_log_returns = result['actuals']
        
        if len(pred_log_returns) == 0:
            continue
        
        true_close = log_return_to_close(true_log_returns, initial_closes[item_id])
        pred_close = log_return_to_close(pred_log_returns, initial_closes[item_id])
        
        val_start = pd.Timestamp(val_start_date_str)
        date_index = pd.date_range(
            start=val_start + pd.Timedelta(days=horizon),
            periods=len(true_log_returns),
            freq='D'
        )
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(date_index, true_close, label="Actual Close", color="black", linewidth=2, marker='o', markersize=3)
        ax.plot(date_index, pred_close, label="Predicted Close", color="tab:orange", linewidth=2, linestyle='--', marker='s', markersize=3, alpha=0.8)
        
        metrics = compute_close_metrics(pred_close, true_close)
        
        textstr = f"MAE: ${metrics['MAE_close']:.2f}\n"
        textstr += f"RMSE: ${metrics['RMSE_close']:.2f}\n"
        textstr += f"MAPE: {metrics['MAPE_close']:.2f}%\n"
        textstr += f"Direction Acc: {metrics['Direction_Accuracy_close']:.2%}"
        
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f"Close Price Prediction: {item_id.upper()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Close Price ($)", fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"close_full_validation_{item_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


def collect_close_metrics_rolling(rolling_results, initial_closes, fold, horizon):
    """
    Rolling prediction 기반 Close metric 집계
    """
    results = []
    
    for result in rolling_results:
        item_id = result['item_id']
        pred_log_returns = result['predictions']
        true_log_returns = result['actuals']
        
        if len(pred_log_returns) == 0 or len(true_log_returns) == 0:
            continue
        
        try:
            true_close = log_return_to_close(true_log_returns, initial_closes[item_id])
            pred_close = log_return_to_close(pred_log_returns, initial_closes[item_id])
            
            metrics = compute_close_metrics(pred_close, true_close)
            metrics.update({
                "item": item_id,
                "fold": fold,
                "horizon": horizon
            })
            
            results.append(metrics)
            
        except Exception as e:
            print(f"    ❌ Error processing {item_id}: {str(e)}")
            continue
    
    return results


# ===============================
# 기존 Metric 함수들
# ===============================
def compute_mae(forecasts, tss):
    maes = []
    for fcst, ts in zip(forecasts, tss):
        y_true = ts.values[-len(fcst.mean):]
        y_pred = fcst.mean
        maes.append(np.mean(np.abs(y_true - y_pred)))
    return float(np.mean(maes))


def directional_accuracy_rolling(rolling_results):
    accs = []
    for result in rolling_results:
        y_true = result['actuals']
        y_pred = result['predictions']
        
        if len(y_true) <= 1:
            continue
        
        sign_true = np.sign(np.diff(y_true))
        sign_pred = np.sign(np.diff(y_pred))
        acc = (sign_true == sign_pred).mean()
        accs.append(acc)
    
    return float(np.mean(accs)) if accs else 0.0


def compute_metrics(forecasts, tss, fold, horizon):
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
    
    metrics = {
        "MAE": compute_mae(forecasts, tss),
        "RMSE": agg_metrics.get("RMSE"),
        "MAPE": agg_metrics.get("MAPE"),
        "fold": fold,
        "horizon": horizon
    }
    
    return metrics


def plot_loss_from_logger(logger, save_path=None):
    csv_file = os.path.join(logger.log_dir, "metrics.csv")
    if not os.path.exists(csv_file):
        return

    df = pd.read_csv(csv_file)

    if "epoch" in df.columns:
        cols = [c for c in ["train_loss", "val_loss"] if c in df.columns]
        if not cols:
            return
        metrics = df.groupby("epoch")[cols].mean().reset_index()
        x = metrics["epoch"]
        xlabel = "Epoch"
    elif "step" in df.columns:
        cols = [c for c in ["train_loss", "val_loss"] if c in df.columns]
        if not cols:
            return
        metrics = df[["step"] + cols].copy()
        x = metrics["step"]
        xlabel = "Training Step"
    else:
        return

    plt.figure(figsize=(10, 5))

    if "train_loss" in metrics and metrics["train_loss"].dropna().any():
        plt.plot(x, metrics["train_loss"], label="Train Loss", marker="o", markersize=3)

    if "val_loss" in metrics and metrics["val_loss"].dropna().any():
        plt.plot(x, metrics["val_loss"], label="Validation Loss", marker="s", markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title("DeepAR Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


# ===============================
# 🚀 Main 함수
# ===============================
def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    cfg.seq_length=20
    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("📁 Loading Dataset")
    print("="*60)
    
    original_dfs = {}
    dfs = {}
    
    for name in ["corn", "wheat", "soybean", "gold", "silver", "copper"]:
        data_path = os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv")
        data = pd.read_csv(data_path)
        data["item_id"] = name
        data['time'] = pd.to_datetime(data['time'])
        
        original_dfs[name] = data.copy()
        dfs[name] = data
        
        print(f"✅ {name}: {len(data)} samples")

    # Feature 추출
    feature_cols = [
        c for c in pd.concat(dfs.values(), ignore_index=True).columns
        if c not in ["time", "item_id", "close"] and not c.startswith("log_return_")
    ]
    
    print(f"\n🔧 Feature columns: {len(feature_cols)}")
    
    # 🧹 Feature Cleaning 추가
    for name in list(dfs.keys()):
        dfs[name] = clean_features(dfs[name], feature_cols)
        dfs[name] = lag_features_by_1day(dfs[name], feature_cols, group_col="item_id", time_col="time")
        dfs[name] = clean_features(dfs[name], feature_cols)  # Lag 후 한번 더 정리
        original_dfs[name] = dfs[name].copy()

    cfg.epochs = 10
    cfg.fold = [0,1,2,3,4,5,6,7] 
    

    all_results_log = []
    all_results_close = []

    for fold in cfg.fold:
        print(f"\n{'='*60}")
        print(f"🔄 Processing Fold {fold}")
        print(f"{'='*60}")
        
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
        
        with open(os.path.join(cfg.data_dir, "rolling_fold.json"), "r") as f:
            fold_data = json.load(f)
        
        val_dates_from_json = fold_data["folds"][fold]["val"]["t_dates"]
        val_start_date_str = str(val_dates_from_json[0])[:10]
        
        initial_closes = {}
        for name in dfs.keys():
            initial_closes[name] = get_initial_close(original_dfs[name], name, val_start_date_str)
        # 모델이 horizon별로 따로 있어. 제가 horzion별로 나눴어 바보죠 
        for h in cfg.horizons: #1 하루치만 가지고 loss 계ㅆ삲 , 5 5일치고 loss 를 계산 , 10일 코딩을 이렇게 짜버렸어.
            #이거 수정 . log_return으로 loss를 예측하는게
            # 모델 하나로 20 일치를 만드는 상황? # log_return_5를 넣은다쳐 #horizon 5 또 5일치를 다음날부터 5일후 5일후 10일까지
            #
            print("\n" + "="*60)
            print(f"🚀 Fold {fold} | Horizon {h}")
            print("="*60)
                # log_return_시점 
            train_ds = build_multi_item_dataset(train_dfs, f"log_return_{h}", feature_cols)
            val_ds = build_multi_item_dataset(val_dfs, f"log_return_{h}", feature_cols)
            
            validate_dataset(train_ds, "Training Dataset")
            validate_dataset(val_ds, "Validation Dataset")
            
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
                lr=1e-4,
                trainer_kwargs={
                    "max_epochs": cfg.epochs,
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "logger": logger,
                    "gradient_clip_val": 1.0,
                },
            )
            
            predictor = estimator.train(
                training_data=train_ds,
                validation_data=val_ds,
            )
            
            rolling_results = rolling_predictions(
    predictor=predictor,
    val_ds=val_ds,
    horizon=h,                 # prediction_length
    context_length=cfg.seq_length,  
    max_steps=None,            # 디버그면 30 같은 값으로
)

            # Metrics
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=val_ds,
                predictor=predictor,
                num_samples=100,
            )
            
            forecasts = list(forecast_it)
            tss = list(ts_it)
            
            metrics = compute_metrics(forecasts, tss, fold, h)
            metrics["Directional_Accuracy"] = directional_accuracy_rolling(rolling_results)
            
            all_results_log.append({
                "item": "ALL",
                "fold": fold,
                "horizon": h,
                **metrics
            })
            
            close_metrics = collect_close_metrics_rolling(rolling_results, initial_closes, fold, h)
            all_results_close.extend(close_metrics)
            
            # 저장
            save_path = Path(cfg.checkpoint_dir) / f"deepar_multi_fold{fold}_h{h}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            predictor.serialize(save_path)
            plot_loss_from_logger(logger, save_path=save_path / "loss_curve.png")
            plot_close_predictions_full(rolling_results, initial_closes, val_start_date_str, h, save_dir=save_path)

    # 최종 저장
    log_df = pd.DataFrame(all_results_log)
    log_csv = Path(cfg.checkpoint_dir) / "deepar_metrics_log_return.csv"
    log_df.to_csv(log_csv, index=False, float_format="%.6f")
    
    if len(all_results_close) > 0:
        close_df = pd.DataFrame(all_results_close)
        close_csv = Path(cfg.checkpoint_dir) / "deepar_metrics_close.csv"
        close_df.to_csv(close_csv, index=False, float_format="%.6f")
        print("\n" + "="*60)
        print("💵 Close Price Metrics Summary")
        print("="*60)
        print(close_df.to_string(index=False))


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)
