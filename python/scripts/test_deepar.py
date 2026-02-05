
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
        df[name] = data
        print(f" {name}: {len(data)} samples")
    
    feature_cols = [
        c for c in pd.concat(df.values(), ignore_index=True).columns
        if c not in ["time", "item_id", "close", "time_idx", "time_idx_int"]
        and not c.startswith("log_return_") 
    ]
    feature_cols = [c for c in feature_cols if c not in SENT_RATIOS and c not in TIME_RATIOS]
    # lag features (1-day shift)
    for name in list(df.keys()):
        df[name] = lag_features_by_1day(df[name], feature_cols, group_col="item_id", time_col="time")
        df[name].replace([np.inf, -np.inf], np.nan, inplace=True) # 다 점수니까
        df[name].fillna(0, inplace=True)
    global_times = sorted(pd.concat(df.values())["time"].unique())
    time2idx = {t: i for i, t in enumerate(global_times)}
    anchor = pd.Timestamp("2000-01-01")
    df["time_idx_int"] = df["time"].map(time2idx)
    df["time_idx"] = anchor + pd.to_timedelta(df["time_idx_int"].astype(int), unit="D")
    
    prediction_length=20
    context_lens=[5,20,60]
    for ctx_len in context_lens:
        train_dfs, test_dfs = {}, {}
        for name, dfi in df.items():
            train_df = dfi.iloc[:-prediction_length].copy()
            test_df  = dfi.copy()   
            train_dfs[name] = train_df
            test_dfs[name]  = test_df
            
        train_dfs[name] = train_df
        test_dfs[name]  = test_df
        estimator = DeepAREstimator(
                    freq="D", prediction_length=prediction_length, context_length=ctx_len,
                    num_feat_dynamic_real=len(feature_cols), num_layers=3, hidden_size=64,
                    dropout_rate=0.1, lr=1e-4, scaling=True, distr_output=StudentTOutput(),
                    trainer_kwargs={
                        "max_epochs": cfg.epochs, "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                        "devices": 1, "logger": logger, "gradient_clip_val": 1.0,
                    },
                )

        predictor = estimator.train(training_data=train_ds)
                
        forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=200
    )# 자동으로 마지막 후행 거기만 하니까 후엥....
        forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=200,
    )

        forecasts = list(forecast_it)
        tss       = list(ts_it)
        base_closes, true_close_list = [], []

        for fcst in forecasts:
            item_id = fcst.item_id
            dfi = df_by_item[item_id]

            # 마지막 예측 시작 시점
            pred_start_ts = fcst.start_date.to_timestamp()

            current_idx = dfi[dfi["time_idx"] >= pred_start_ts].index[0]

            base_close = float(dfi.loc[current_idx - 1, "close"])
            true_close = dfi.loc[current_idx : current_idx + prediction_length - 1, "close"].values

            base_closes.append(base_close)
            true_close_list.append(true_close)
            
        price_q = {0.1: [], 0.3: [], 0.5: [], 0.7: [], 0.9: []}

        for i, fcst in enumerate(forecasts):
            log_rel = np.cumsum(fcst.samples, axis=1)
            price_paths = base_closes[i] * np.exp(log_rel)

            for q in price_q:
                price_q[q].append(np.quantile(price_paths, q, axis=0))
                
        pd.DataFrame(rows).to_csv("test_forecast_close_quantiles.csv", index=False)


            



            
        