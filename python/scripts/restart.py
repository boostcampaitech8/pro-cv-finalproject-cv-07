import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
from pathlib import Path
from gluonts.torch.distributions import NormalOutput , StudentTOutput
PROJECT_ROOT = Path("/data/ephemeral/home/pro-cv-finalproject-cv-07/python")
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset import build_multi_item_dataset, deepar_split, lag_features_by_1day

import tyro
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from pytorch_lightning.loggers import CSVLogger
from gluonts.dataset.split import split 
def build_rolling_instances(entries, ctx_len, prediction_length, item_id_key="item_id"):
    inst_list = []
    for entry in entries:
        target = np.asarray(entry["target"]).astype(float)
        total_len = target.shape[-1]
        
        # 동적 피처가 있는지 확인 (DeepAR 필수 요소)
        dyn_feat = entry.get("feat_dynamic_real", None)
        
        first_start = ctx_len
        last_start = total_len - prediction_length
        
        if last_start < first_start:
            continue

        start_date = entry.get("start", None)
        item_id = entry.get(item_id_key, entry.get("item_id", None))

        for s in range(first_start, last_start + 1):
            # 1. target 슬라이싱 (예측 지점 s까지의 과거 + 예측할 미래 prediction_length)
            inst_target = target[: s + prediction_length].copy()
            
            inst = {
                "target": inst_target,
            }
            
            # 2. 🔥 feat_dynamic_real 슬라이싱 (중요!)
            # target과 동일한 길이만큼 피처도 잘라서 넣어줍니다.
            if dyn_feat is not None:
                # dyn_feat 구조: (feature_dim, total_len)
                inst["feat_dynamic_real"] = dyn_feat[:, : s + prediction_length].copy()
            
            # 3. 메타데이터 보존
            if start_date is not None:
                inst["start"] = start_date
            if item_id is not None:
                inst["item_id"] = item_id
                
            inst_list.append(inst)
            
    return inst_list
# Metrics helpers
# -------------------------
def directional_accuracy(true_close, pred_close, base_close=None):
    true_diff = np.diff(true_close)
    pred_diff = np.diff(pred_close)
    mask = np.isfinite(true_diff) & np.isfinite(pred_diff)
    if mask.sum() == 0: return np.nan
    return np.mean(np.sign(true_diff[mask]) == np.sign(pred_diff[mask]))

def pinball_loss(y, qhat, q):
    y, qhat = np.asarray(y, dtype=float), np.asarray(qhat, dtype=float)
    e = y - qhat
    mask = np.isfinite(e)
    if mask.sum() == 0: return np.nan
    e = e[mask]
    return np.mean(np.maximum(q * e, (q - 1) * e))

# -------------------------
# Plot helpers
# -------------------------
def plot_forecasts_per_item_return(tss, forecasts, save_dir: Path, prediction_length: int, max_items: int = None):
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(tss) if max_items is None else min(len(tss), max_items)
    for i in range(0, min(n, prediction_length * 3), prediction_length):
        ts, fcst = tss[i], forecasts[i]
        item_id = getattr(fcst, "item_id", None)
        pred_start = fcst.start_date.to_timestamp() if hasattr(fcst.start_date, "to_timestamp") else fcst.start_date
        plt.figure(figsize=(12, 5))
        ts.iloc[-prediction_length:].plot(label="Actual (return)", color="black", linewidth=1.5)
        fcst.plot(color="tab:blue")
        plt.axvline(pred_start, color="red", linestyle="--", linewidth=1, label="Prediction start")
        plt.title(f"{item_id} | Return-space: Actual vs Forecast")
        plt.legend(); plt.tight_layout()
        plt.savefig(save_dir / f"{item_id}_return{i}_forecast.png", dpi=150); plt.close()

def plot_close_forecasts_per_item_steps(forecasts, true_close_list, price_q10_list,price_q30_list, price_q50_list,price_q70_list, price_q90_list, save_dir: Path, max_items: int = None):
    save_dir.mkdir(parents=True, exist_ok=True)
    predict_length=20
    n = len(forecasts) if max_items is None else min(len(forecasts), max_items)
    for i in range(0, min(n, predict_length * 3), predict_length):
        fcst, item_id = forecasts[i], getattr(forecasts[i], "item_id", None)
        H = fcst.prediction_length
        x = np.arange(H)
        plt.figure(figsize=(12, 5))
        plt.plot(x, true_close_list[i], color="black", linewidth=1.5, label="True close")
        plt.plot(x, price_q50_list[i], color="tab:blue", linewidth=1.5, label="Pred close (p50)")
        plt.fill_between(x, price_q10_list[i], price_q90_list[i], color="tab:blue", alpha=0.2, label="p10-p90")
        plt.fill_between(x, price_q30_list[i], price_q70_list[i], color="tab:red", alpha=0.2, label="p10-p90")
        plt.title(f"{item_id} | Close-space forecast"); plt.legend(); plt.tight_layout()
        plt.savefig(save_dir / f"{item_id}_close{i}_forecast.png", dpi=150); plt.close()

# -------------------------
# Metrics Calculation
# -------------------------
def compute_all_metrics(forecasts, tss, true_close_list, price_q50_list, base_closes, prediction_length, scale=1.0):
    all_rows = []
    for i in range(len(forecasts)):
        fc, item_id = forecasts[i], getattr(forecasts[i], "item_id", None)
        ts_actual = tss[i].values.astype(float).flatten() / scale
        fc_samples = fc.samples / scale
        pm = fc_samples.mean(axis=0)
        
        # Distribution Diagnosis
        true_std, pred_mean_std = float(ts_actual.std()), float(pm.std())
        ratio = true_std / (pred_mean_std + 1e-12)
        
        # Return metrics
        y_true_ret, y_pred_ret = tss[i].iloc[-prediction_length:].values.flatten(), fc.quantile(0.5)
        return_da = float(np.mean(np.sign(y_true_ret) == np.sign(y_pred_ret)))
        
        # Close metrics
        y_true_close, p50_close = true_close_list[i], price_q50_list[i]
        close_mae = float(np.mean(np.abs(p50_close - y_true_close)))
        
        all_rows.append({
            "item_id": item_id, "ratio": ratio, "return_da": return_da, "close_mae": close_mae,
            "pred_sigma_avg": float(fc_samples.std(axis=0).mean()),"true_sigma_avg":true_std
        })
    print(f"DEBUG: len(forecasts)={len(forecasts)}, len(tss)={len(tss)}, len(true_close_list)={len(true_close_list)}")
    return pd.DataFrame(all_rows)

def print_metrics_summary(metrics_df, ctx_len, fold):
    print(f"\n📊 Summary (Fold {fold}, Ctx {ctx_len}): Ratio Avg={metrics_df['ratio'].mean():.3f}, Return DA={metrics_df['return_da'].mean():.3f}")

def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📁 Loading Dataset")
    print("=" * 60)
    SCALE = 1
    df = {}
    SENT_SCORES = ["sentiment_score_mean","sentiment_score_std","sentiment_score_max","sentiment_score_min"]
    TIME_SCORES = ["timeframe_score_mean","timeframe_score_std","timeframe_score_max","timeframe_score_min"]
    SENT_RATIOS = ["sentiment_neg_ratio","sentiment_neu_ratio","sentiment_pos_ratio"]
    TIME_RATIOS = ["time_past_ratio","time_present_ratio","time_future_ratio"]

    for name in ["corn", "wheat", "soybean", "gold", "silver", "copper"]:
        data_path = os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv")
        data = pd.read_csv(data_path)
        data["item_id"] = name
        data["time"] = pd.to_datetime(data["time"])
        data["rv_5"] = data["log_return_1"].rolling(5).std()
        data["rv_10"] = data["log_return_1"].rolling(10).std()
        for features in ['open','high',"EMA",'EMA_5',"EMA_10"]:
            data[features]=data[features] / data["close"]
        df[name] = data
        print(data['rv_5'])
        print(f" {name}: {len(data)} samples")
    
    feature_cols = [
        c for c in pd.concat(df.values(), ignore_index=True).columns
        if c not in ["time", "item_id", "close", "time_idx", "time_idx_int","EMA_20","EMA_50",'EMA_100','Volume']
        and not c.startswith("log_return_") 
    ]
    feature_cols = [c for c in feature_cols if c not in SENT_RATIOS and c not in TIME_RATIOS]
    feature_cols = [c for c in feature_cols if c not in SENT_SCORES and c not in TIME_SCORES] 
    #feature_cols += ["rv_5", "rv_10"]#, "abs_ret"]df["abs_ret"] = np.abs(df["ret"])
    # lag features (1-day shift)
    # for name in list(df.keys()):
    #     df[name] = lag_features_by_1day(df[name], feature_cols, group_col="item_id", time_col="time")
    #     df[name].replace([np.inf, -np.inf], np.nan, inplace=True) # 다 점수니까
    #     df[name].fillna(0, inplace=True)

    cfg.epochs = 10
    cfg.fold = [0,1,2,3,4,5,6,7]
    seq_lengths = [5, 20, 60]
    prediction_length = 20
#학습 할 때 가중치가 바뀌지 . 모델이 튜닝 . h1 h5 h20 
    for fold in cfg.fold:
        print(f"\n🔄 Processing Fold {fold}")

        train_dfs, val_dfs = {}, {}
        all_item_list = []
        anchor = pd.Timestamp("2000-01-01")
        global_times = sorted(pd.concat(df.values())["time"].unique())
        time2idx = {t: i for i, t in enumerate(global_times)}
        for name, dfi in df.items():
            dfi["time_idx_int"] = dfi["time"].map(time2idx)
            dfi["time_idx"] = anchor + pd.to_timedelta(dfi["time_idx_int"].astype(int), unit="D")


            
            
        for name, dfi in df.items():
            
            train_df, val_df = deepar_split(dfi, os.path.join(PROJECT_ROOT, "src/datasets/rolling_fold.json"), fold)
            train_df = train_df.sort_values("time").reset_index(drop=True).copy()
            val_df = val_df.sort_values("time").reset_index(drop=True).copy()
            train_dfs[name], val_dfs[name] = train_df, val_df
            all_item_list.append(pd.concat([train_df, val_df], axis=0))
        
        df_by_item = { name: pd.concat([train_dfs[name], val_dfs[name]], axis=0, ignore_index=True).sort_values("time_idx").copy() for name in df.keys() }
        train_ds = build_multi_item_dataset(train_dfs, "log_return_1", feature_cols)
        train_list = list(train_ds)
        val_ds = build_multi_item_dataset(val_dfs, "log_return_1", feature_cols)
        val_list = list(val_ds)

        for ctx_len in seq_lengths:
            print(f"\n🚀 Running for Context Length: {ctx_len}")
            logger = CSVLogger(save_dir=cfg.checkpoint_dir, name=f"fold{fold}_ctx{ctx_len}")
            run_dir = Path(logger.log_dir)
            plots_dir, metrics_dir = run_dir / "plots", run_dir / "metrics"
            plots_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)

            # 1. Validation with context (Context + Val)
            val_with_context_dfs = {}
            for name in df.keys():
                context_df = train_dfs[name].tail(ctx_len) 
                combined_df = pd.concat([context_df, val_dfs[name]], axis=0).reset_index(drop=True)
                val_with_context_dfs[name] = combined_df

            val_ds_with_ctx = build_multi_item_dataset(val_with_context_dfs, "log_return_1", feature_cols)
            val_list_with_ctx = list(val_ds_with_ctx)

            rolling_val_list = build_rolling_instances(val_list_with_ctx, ctx_len=ctx_len, prediction_length=prediction_length)

            print(f"✅ 안전하게 생성된 rolling windows: {len(rolling_val_list)}")

            # 3. Estimator & Training
            estimator = DeepAREstimator(
                freq="D", prediction_length=prediction_length, context_length=ctx_len,
                num_feat_dynamic_real=len(feature_cols), num_layers=3, hidden_size=64,
                dropout_rate=0.1, lr=1e-4, scaling=False, distr_output=StudentTOutput(),
                trainer_kwargs={
                    "max_epochs": cfg.epochs, "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1, "logger": logger, "gradient_clip_val": 1.0,
                },
            )

            predictor = estimator.train(training_data=train_ds, validation_data=val_ds)

            # 4. Evaluation
            forecast_it, ts_it = make_evaluation_predictions(dataset=rolling_val_list, predictor=predictor, num_samples=200)
            forecasts, tss = list(forecast_it), list(ts_it)
 
            # 5. Price-Space Reconstruction (Mapping back to 'close')
            base_closes, true_close_list, valid_indices = [], [], []
            for i, fcst in enumerate(forecasts):
                item_id = str(getattr(fcst, "item_id", "")).strip()
                pred_start_ts = fcst.start_date.to_timestamp()
                dfi = df_by_item.get(item_id)
                if dfi is None or dfi.empty:
                    continue

                # pred_start_ts를 tz-naive Timestamp로 통일
                if hasattr(fcst.start_date, "to_timestamp"):
                    pred_start_ts = fcst.start_date.to_timestamp()
                else:
                    pred_start_ts = pd.Timestamp(fcst.start_date)
                if getattr(pred_start_ts, "tz", None) is not None:
                    pred_start_ts = pred_start_ts.tz_localize(None)

                # time_idx와 pred를 둘 다 datetime64[ns]로 통일
                times = pd.to_datetime(dfi["time_idx"], utc=False).to_numpy(dtype="datetime64[ns]")
                pred = np.datetime64(pd.Timestamp(pred_start_ts).to_datetime64())
#분포 , 형한테서 output .csv 값 만들고 다른 에서 concat . plot도 같이하는게  따로 저장한다고 
# 색깔? 이런걸로 구분? / 같이 이미지 있는게 아니다..? df . model 하나# 모델별로 csv로 저장하고/ 합치는 코드가 있어야되잖아./합치는 거를 또 저장/ 
# vlm 어찌됐든 주는거는 그냥 plot해서 내 분포 그리고 형 예측값 ...... 이미지로 쓸꺼잖아 .# 양식 통일 VLM 값도
#근데 내 기억으로는 close로 하기로 했던 것같은데... 근데 약간 참고지표 느낌이잖아그거는 뭐 해보ㅈ
                # 정렬/매칭 확인
                pos = int(np.searchsorted(times, pred))
                if pos >= len(times) or times[pos] != pred:
                    # 정확 매칭이 없으면 flatnonzero로 한 번 더 시도
                    mask = pd.to_datetime(dfi["time_idx"], utc=False).eq(pd.Timestamp(pred_start_ts))
                    if not mask.any():
                        continue
                    pos = int(np.flatnonzero(mask.to_numpy())[0])

                end = pos + prediction_length
                if end > len(dfi):
                    continue

                base_close = float(dfi["close"].iloc[pos])
                true_close = dfi["close"].iloc[pos:end].to_numpy()

                base_closes.append(base_close)
                true_close_list.append(true_close)
                valid_indices.append(i)

            
            forecasts = [forecasts[i] for i in valid_indices]
            tss = [tss[i] for i in valid_indices]
            print("forecasts",forecasts)
            
            # Price Path Calculation
            price_q10_list, price_q30_list , price_q50_list, price_q70_list,price_q90_list = [], [], [] ,[] , []
            for i, fcst in enumerate(forecasts):
                log_rel = np.cumsum(fcst.samples / SCALE, axis=1)
                price_paths = base_closes[i] * np.exp(log_rel)
                price_q10_list.append(np.quantile(price_paths, 0.1, axis=0))
                price_q30_list.append(np.quantile(price_paths, 0.3, axis=0))
                price_q50_list.append(np.quantile(price_paths, 0.5, axis=0))
                price_q70_list.append(np.quantile(price_paths, 0.7, axis=0))
                price_q90_list.append(np.quantile(price_paths, 0.9, axis=0))

            # 6. Final Metrics & Plot
            metrics_df = compute_all_metrics(forecasts, tss, true_close_list, price_q50_list, base_closes, prediction_length, SCALE)
            metrics_df.to_csv(metrics_dir / "all_metrics.csv", index=False)
            
            NUM_PLOT=3
            plot_forecasts_per_item_return(tss[-NUM_PLOT:], forecasts[-NUM_PLOT:], plots_dir    , prediction_length)
            plot_close_forecasts_per_item_steps(forecasts[-NUM_PLOT:], true_close_list[-NUM_PLOT:], price_q10_list[-NUM_PLOT:],price_q30_list[-NUM_PLOT:], price_q50_list[-NUM_PLOT:],price_q70_list[-NUM_PLOT:], price_q90_list[-NUM_PLOT:], plots_dir)

if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)