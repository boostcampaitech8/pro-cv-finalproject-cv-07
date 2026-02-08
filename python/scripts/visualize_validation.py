"""
Validation Set Visualization Tool (100% Feature Engineering 일치)
✅ quantile_loss True/False 자동 분기 처리

Feature Engineering 정의:
  log_return_h = log(close.shift(-(h-1)) / close.shift(1))
               = log(close_{t+h-1} / close_{t-1})
  
  즉, t-1 시점의 log_return_h는:
    close_{t+h-1} / close_{t-1} 비율

Close 역계산:
  close_{t+h-1} = close_{t-1} * exp(log_return_h)

Usage:
    python visualize_validation.py --target_commodity corn --fold 0
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import tyro

sys.path.append(str(Path(__file__).parent.parent))

from src.configs.train_config import TrainConfig
from src.data.dataset_tft import TFTDataLoader
from src.models.TFT import TemporalFusionTransformer

class SimpleYScaler:
    '''y_scaler를 로드해서 inverse_transform 사용'''
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std
    
    def inverse_transform(self, Y_scaled: np.ndarray) -> np.ndarray:
        return (Y_scaled * self.std + self.mean).astype(np.float32)


def get_target_date_for_horizon(base_date_str, price_df, horizon):
    """
    Feature engineering 정의에 맞춰 target date 계산
    
    log_return_h @ t-1 = log(close_{t+h-1} / close_{t-1})
    
    따라서:
    - base_date = t-1
    - target_date = t-1 + h = t+h-1
    
    Args:
        base_date_str: "YYYY-MM-DD" (t-1 시점)
        price_df: 가격 데이터
        horizon: 1, 5, 10, 20
    
    Returns:
        target_date_str: "YYYY-MM-DD" (t+h-1 시점)
    """
    trading_dates = pd.to_datetime(price_df['time']).dt.date.tolist()
    base_date = pd.to_datetime(base_date_str).date()
    
    if base_date not in trading_dates:
        return None
    
    base_idx = trading_dates.index(base_date)
    target_idx = base_idx + horizon  # t-1 + h = t+h-1
    
    if target_idx >= len(trading_dates):
        return None
    
    return str(trading_dates[target_idx])


def log_return_to_close(base_date_str, log_return, price_df):
    """
    Feature engineering과 100% 일치하는 close 역계산
    
    Feature engineering:
      log_return_h @ t-1 = log(close_{t+h-1} / close_{t-1})
    
    따라서:
      close_{t+h-1} = close_{t-1} * exp(log_return_h)
    
    Args:
        base_date_str: "YYYY-MM-DD" (t-1 시점)
        log_return: float (예측값)
        price_df: 가격 데이터
    
    Returns:
        close_target: float (t+h-1 시점 close)
    """
    price_df_copy = price_df.copy()
    price_df_copy['date_str'] = pd.to_datetime(price_df_copy['time']).dt.strftime('%Y-%m-%d')
    
    # t-1 시점의 close 찾기
    base_row = price_df_copy[price_df_copy['date_str'] == base_date_str]
    
    if len(base_row) == 0:
        return np.nan
    
    base_close = base_row['close'].values[0]  # close_{t-1}
    
    # close_{t+h-1} = close_{t-1} * exp(log_return_h)
    close_target = base_close * np.exp(log_return)
    
    return close_target


def plot_single_horizon_quantile(
    dates,
    true_log, pred_q10, pred_q50, pred_q90,
    true_close, close_q10, close_q50, close_q90,
    horizon,
    save_path
):
    """단일 horizon 시각화 (Quantile 버전)"""
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Log Return
    ax = axes[0]
    ax.plot(dates_dt, true_log, label="True", color="black", linewidth=1.5)
    ax.plot(dates_dt, pred_q50, label="Pred (Median)", color="blue", linewidth=1.5)
    ax.fill_between(dates_dt, pred_q10, pred_q90, alpha=0.25, color='blue', label='80% Interval')
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"H{horizon} Log Return (Validation)", fontsize=14, fontweight='bold')
    ax.set_ylabel('Log Return', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # MAE, RMSE, DA
    mae = np.mean(np.abs(true_log - pred_q50))
    rmse = np.sqrt(np.mean((true_log - pred_q50) ** 2))
    da = np.mean((true_log > 0) == (pred_q50 > 0)) * 100
    
    ax.text(0.02, 0.98, f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nDA: {da:.2f}%',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Close
    ax = axes[1]
    ax.plot(dates_dt, true_close, label="True Close", color="black", linewidth=1.5)
    ax.plot(dates_dt, close_q50, label="Pred Close (Median)", color="red", linewidth=1.5)
    ax.fill_between(dates_dt, close_q10, close_q90, alpha=0.25, color='red', label='80% Interval')
    ax.set_title(f"H{horizon} Close Price (Validation)", fontsize=14, fontweight='bold')
    ax.set_ylabel('Close Price', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    
    # Close MAE, RMSE
    close_mae = np.mean(np.abs(true_close - close_q50))
    close_rmse = np.sqrt(np.mean((true_close - close_q50) ** 2))
    
    ax.text(0.02, 0.98, f'MAE: {close_mae:.4f}\nRMSE: {close_rmse:.4f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_single_horizon_point(
    dates,
    true_log, pred_log,
    true_close, pred_close,
    horizon,
    save_path
):
    """단일 horizon 시각화 (Point Prediction 버전)"""
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Log Return
    ax = axes[0]
    ax.plot(dates_dt, true_log, label="True", color="black", linewidth=1.5)
    ax.plot(dates_dt, pred_log, label="Prediction", color="blue", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"H{horizon} Log Return (Validation)", fontsize=14, fontweight='bold')
    ax.set_ylabel('Log Return', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # MAE, RMSE, DA
    mae = np.mean(np.abs(true_log - pred_log))
    rmse = np.sqrt(np.mean((true_log - pred_log) ** 2))
    da = np.mean((true_log > 0) == (pred_log > 0)) * 100
    
    ax.text(0.02, 0.98, f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nDA: {da:.2f}%',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Close
    ax = axes[1]
    ax.plot(dates_dt, true_close, label="True Close", color="black", linewidth=1.5)
    ax.plot(dates_dt, pred_close, label="Prediction", color="red", linewidth=1.5)
    ax.set_title(f"H{horizon} Close Price (Validation)", fontsize=14, fontweight='bold')
    ax.set_ylabel('Close Price', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    
    # Close MAE, RMSE
    close_mae = np.mean(np.abs(true_close - pred_close))
    close_rmse = np.sqrt(np.mean((true_close - pred_close) ** 2))
    
    ax.text(0.02, 0.98, f'MAE: {close_mae:.4f}\nRMSE: {close_rmse:.4f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main(config: TrainConfig):
    fold = config.fold[0]
    h_tag = "h" + "-".join(map(str, config.horizons))

    fold_dir = Path(config.checkpoint_dir) / f"TFT_{config.target_commodity}_fold{fold}_{h_tag}"
    scaler_path = fold_dir / "scaler.npz"
    y_scaler = None

    if scaler_path.exists():
        scaler_data = np.load(scaler_path, allow_pickle=True)
        
        if scaler_data.get("scale_y", False):
            if "y_scaler_mean" in scaler_data:
                y_mean = scaler_data["y_scaler_mean"]
                y_std = scaler_data["y_scaler_std"]
            else:
                y_mean = scaler_data["y_mean"]
                y_std = scaler_data["y_std"]
            
            y_scaler = SimpleYScaler(y_mean, y_std)
            print(f"✓ Loaded Y scaler")

    viz_dir = fold_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"Validation Visualization")
    print(f"{'='*60}")
    print(f"Commodity: {config.target_commodity}")
    print(f"Fold: {fold}")
    print(f"Horizons: {config.horizons}")
    print(f"Quantile Loss: {config.quantile_loss}")  # ← 추가!
    print(f"{'='*60}\n")

    # 가격 데이터 로드
    price_path = os.path.join(
        config.data_dir,
        f"preprocessing/{config.target_commodity}_feature_engineering.csv"
    )
    price_df = pd.read_csv(price_path)
    price_df['time'] = pd.to_datetime(price_df['time'])

    # Data loader
    data_loader = TFTDataLoader(
        price_data_path=price_path,
        news_data_path=os.path.join(config.data_dir, "news_features.csv"),
        split_file=os.path.join(config.data_dir, "rolling_fold.json"),
        seq_length=config.seq_length,
        horizons=config.horizons,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )

    _, valid_loader, validT = data_loader.get_fold_loaders(fold)
    base_dates = [str(t)[:10] for t in validT]
    print(f"✓ Base dates (t-1): {len(base_dates)}")
    print(f"  Range: {base_dates[0]} ~ {base_dates[-1]}")

    # 모델 로드
    checkpoint = torch.load(fold_dir / "best_model.pt", map_location="cpu")

    # ===================================================================
    # ✅ quantile_loss에 따라 quantiles 파라미터 설정
    # ===================================================================
    if config.quantile_loss:
        quantiles = config.quantiles
        print(f"✓ Mode: Quantile Regression (quantiles={quantiles})")
    else:
        quantiles = None
        print(f"✓ Mode: Point Prediction (MSELoss)")
    # ===================================================================

    model = TemporalFusionTransformer(
        num_features=data_loader.X.shape[-1],
        num_horizons=len(config.horizons),
        hidden_dim=checkpoint["config"]["hidden_dim"],
        lstm_layers=checkpoint["config"]["num_layers"],
        attention_heads=checkpoint["config"]["attention_heads"],
        dropout=checkpoint["config"]["dropout"],
        use_variable_selection=config.use_variable_selection,
        quantiles=quantiles,
        news_projection_dim=config.news_projection_dim
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n🔮 Predicting...")
    preds, trues = [], []
    with torch.no_grad():
        for X, Y in valid_loader:
            out = model(X)
            preds.append(out["predictions"].numpy())
            trues.append(Y.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # ✅ Inverse scale Y for visualization/metrics
    if y_scaler is not None:
        print("\n🔄 Applying inverse scaling...")
        trues = y_scaler.inverse_transform(trues)

        if config.quantile_loss:
            # preds: [N,H,Q]
            N, H, Q = preds.shape
            preds_scaled = []
            for q in range(Q):
                pred_q = preds[:, :, q]  # [N, H]
                pred_q_orig = y_scaler.inverse_transform(pred_q)
                preds_scaled.append(pred_q_orig)
            preds = np.stack(preds_scaled, axis=-1)  # [N, H, Q]
            print(f"  ✓ Scaled back quantile predictions: {preds.shape}")
        else:
            # preds: [N,H]
            preds = y_scaler.inverse_transform(preds)
            print(f"  ✓ Scaled back point predictions: {preds.shape}")
    else:
        print("ℹ️  No Y scaling - predictions in original scale")

    # ===================================================================
    # ✅ 차원 체크 및 자동 처리
    # ===================================================================
    if config.quantile_loss:
        # Quantile mode: [N, H, Q]
        print(f"✓ Predictions: {preds.shape} (Quantile mode)")
        print(f"✓ Targets: {trues.shape}")
        assert preds.ndim == 3, f"Expected 3D predictions, got {preds.ndim}D"
    else:
        # Point prediction mode: [N, H]
        print(f"✓ Predictions: {preds.shape} (Point prediction mode)")
        print(f"✓ Targets: {trues.shape}")
        assert preds.ndim == 2, f"Expected 2D predictions, got {preds.ndim}D"
    # ===================================================================

    # ===== CSV 데이터 수집 =====
    csv_rows = []

    # ===== Horizon별 시각화 =====
    for h_idx, horizon in enumerate(config.horizons):
        print(f"\n📊 Processing H{horizon}...")
        
        # Target dates 계산
        target_dates = []
        valid_indices = []
        
        for i, base_date in enumerate(base_dates):
            target_date = get_target_date_for_horizon(base_date, price_df, horizon)
            if target_date is not None:
                target_dates.append(target_date)
                valid_indices.append(i)
        
        print(f"  ✓ Valid samples: {len(valid_indices)} / {len(base_dates)}")
        
        if len(valid_indices) == 0:
            print(f"  ⚠️  No valid samples for H{horizon}, skipping...")
            continue
        
        # ===================================================================
        # ✅ 예측값 추출 (Quantile vs Point)
        # ===================================================================
        valid_indices = np.array(valid_indices)
        true_log = trues[valid_indices, h_idx]
        
        if config.quantile_loss:
            # Quantile mode: [N, H, Q]
            q10_idx = config.quantiles.index(0.1)
            q50_idx = config.quantiles.index(0.5)
            q90_idx = config.quantiles.index(0.9)

            q10 = preds[valid_indices, h_idx, q10_idx]
            q50 = preds[valid_indices, h_idx, q50_idx]
            q90 = preds[valid_indices, h_idx, q90_idx]
            pred_log = q50  # Median
        else:
            # Point prediction mode: [N, H]
            pred_log = preds[valid_indices, h_idx]
            q10 = None
            q50 = pred_log
            q90 = None
        # ===================================================================
        
        # Close 변환
        valid_base_dates = [base_dates[i] for i in valid_indices]
        
        true_close = []
        pred_close = []
        close_q10 = []
        close_q90 = []
        
        for bd, tl, pl in zip(valid_base_dates, true_log, pred_log):
            # Feature engineering과 일치:
            # close_{t+h-1} = close_{t-1} * exp(log_return_h)
            true_close.append(log_return_to_close(bd, tl, price_df))
            pred_close.append(log_return_to_close(bd, pl, price_df))
            
            if config.quantile_loss:
                close_q10.append(log_return_to_close(bd, q10[len(close_q10)], price_df))
                close_q90.append(log_return_to_close(bd, q90[len(close_q90)], price_df))
        
        true_close = np.array(true_close)
        pred_close = np.array(pred_close)
        
        if config.quantile_loss:
            close_q10 = np.array(close_q10)
            close_q90 = np.array(close_q90)
        
        # Metrics
        mae = np.mean(np.abs(true_log - pred_log))
        rmse = np.sqrt(np.mean((true_log - pred_log) ** 2))
        da = np.mean((true_log > 0) == (pred_log > 0)) * 100
        
        print(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}, DA: {da:.2f}%")
        
        # ===================================================================
        # ✅ 시각화 (Quantile vs Point)
        # ===================================================================
        if config.quantile_loss:
            plot_single_horizon_quantile(
                target_dates,
                true_log, q10, q50, q90,
                true_close, close_q10, pred_close, close_q90,
                horizon,
                viz_dir / f"validation_h{horizon}.png"
            )
        else:
            plot_single_horizon_point(
                target_dates,
                true_log, pred_log,
                true_close, pred_close,
                horizon,
                viz_dir / f"validation_h{horizon}.png"
            )
        # ===================================================================
        print(f"  ✓ Saved: validation_h{horizon}.png")
        
        # ===================================================================
        # ✅ CSV 데이터 수집 (Quantile vs Point)
        # ===================================================================
        if config.quantile_loss:
            # Quantile mode
            for bd, td, tl, p10, p50, p90, tc, c10, c50, c90 in zip(
                valid_base_dates, target_dates,
                true_log, q10, q50, q90,
                true_close, close_q10, pred_close, close_q90
            ):
                csv_rows.append({
                    "base_date": bd,
                    "target_date": td,
                    "horizon": horizon,
                    "true_log_return": float(tl),
                    "pred_log_q10": float(p10),
                    "pred_log_q50": float(p50),
                    "pred_log_q90": float(p90),
                    "true_close": float(tc),
                    "pred_close_q10": float(c10),
                    "pred_close_q50": float(c50),
                    "pred_close_q90": float(c90),
                })
        else:
            # Point prediction mode
            for bd, td, tl, pl, tc, pc in zip(
                valid_base_dates, target_dates,
                true_log, pred_log,
                true_close, pred_close
            ):
                csv_rows.append({
                    "base_date": bd,
                    "target_date": td,
                    "horizon": horizon,
                    "true_log_return": float(tl),
                    "pred_log_return": float(pl),
                    "true_close": float(tc),
                    "pred_close": float(pc),
                })
        # ===================================================================

    # ===== CSV 저장 =====
    if csv_rows:
        df_csv = pd.DataFrame(csv_rows)
        
        # 정렬: base_date → horizon 순서
        df_csv = df_csv.sort_values(['base_date', 'horizon']).reset_index(drop=True)
        
        csv_path = fold_dir / "validation_predictions.csv"
        df_csv.to_csv(csv_path, index=False)
        
        print(f"\n💾 CSV saved:")
        print(f"  Path: {csv_path}")
        print(f"  Rows: {len(df_csv)}")
        print(f"  Columns: {list(df_csv.columns)}")
        
        # Sample 출력
        print(f"\n📋 Sample (first 5 rows):")
        print(df_csv.head())

    print(f"\n{'='*60}")
    print("✅ Visualization completed!")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  📁 Visualizations: {viz_dir}/")
    for h in config.horizons:
        print(f"      - validation_h{h}.png")
    print(f"  📁 CSV: {fold_dir}/validation_predictions.csv")
    
    if config.quantile_loss:
        print(f"\n  Mode: Quantile Regression (q10, q50, q90)")
    else:
        print(f"\n  Mode: Point Prediction (single value)")
    print()


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)
