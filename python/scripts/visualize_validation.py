"""
Validation Set Visualization Tool - Fixed

Usage:
    python visualize_validation.py --target_commodity corn --fold 0
    
저장 위치: checkpoints/TFT_corn_fold0/visualizations/
"""

import os
import sys
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


def log_return_to_close(dates, log_returns, price_data_path, horizons):
    """log_return → close price 변환"""
    df = pd.read_csv(price_data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    close_prices = np.zeros_like(log_returns)
    
    for i, date in enumerate(dates):
        date_obj = pd.to_datetime(str(date)[:10])
        matching_rows = df[df['time'] == date_obj]
        
        if len(matching_rows) == 0:
            continue
        
        base_close = matching_rows['close'].values[0]
        
        for h_idx in range(len(horizons)):
            close_prices[i, h_idx] = base_close * np.exp(log_returns[i, h_idx])
    
    return close_prices


def plot_single_horizon(
    dates, true_log, pred_log, true_close, pred_close,
    horizon, save_path
):
    """
    단일 horizon 시각화 (2x2 subplot)
    - log_return 예측
    - log_return error
    - close 예측
    - close error
    """
    dates_dt = [datetime.strptime(str(d)[:10], '%Y-%m-%d') for d in dates]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # ===== Log Return =====
    ax = axes[0, 0]
    ax.plot(dates_dt, true_log, label='True', color='blue', linewidth=1.5)
    ax.plot(dates_dt, pred_log, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Log Return', fontsize=11)
    ax.set_title(f'Validation - H{horizon} - Log Return', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # ===== Log Return Error =====
    ax = axes[0, 1]
    residuals = true_log - pred_log
    ax.plot(dates_dt, residuals, color='green', linewidth=1, alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.fill_between(dates_dt, residuals, 0, alpha=0.3, color='green')
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Log Return Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    ax.text(0.02, 0.98, f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== Close Price =====
    ax = axes[1, 0]
    ax.plot(dates_dt, true_close, label='True', color='blue', linewidth=1.5)
    ax.plot(dates_dt, pred_close, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Close Price', fontsize=11)
    ax.set_title('Close Price Prediction', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # ===== Close Price Error =====
    ax = axes[1, 1]
    close_error = true_close - pred_close
    ax.plot(dates_dt, close_error, color='purple', linewidth=1, alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.fill_between(dates_dt, close_error, 0, alpha=0.3, color='purple')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Close Price Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    close_mae = np.mean(np.abs(close_error))
    close_rmse = np.sqrt(np.mean(close_error**2))
    ax.text(0.02, 0.98, f'MAE: {close_mae:.4f}\nRMSE: {close_rmse:.4f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(config: TrainConfig):
    """메인 함수"""
    
    print(f"\n{'='*60}")
    print(f"Validation Set Visualization")
    print(f"{'='*60}")
    print(f"Target: {config.target_commodity}")
    print(f"Fold: {config.fold[0]}")
    print(f"{'='*60}\n")

    h_tag = "h" + "-".join(map(str, config.horizons))
    print(f"[DEBUG] horizons = {config.horizons} -> h_tag = {h_tag}")
    
    if len(config.fold) != 1:
        raise ValueError("Please specify single fold. e.g., --fold 0")
    
    fold_idx = config.fold[0]
    
    # 경로 설정
    price_file = f"preprocessing/{config.target_commodity}_feature_engineering.csv"
    news_file = "news_features.csv"
    
    price_path = os.path.join(config.data_dir, price_file)
    news_path = os.path.join(config.data_dir, news_file)
    split_file = os.path.join(config.data_dir, "rolling_fold.json")
    
    # ===== Checkpoint 경로 (수정됨!) =====
    fold_dir = Path(config.checkpoint_dir) / f"TFT_{config.target_commodity}_fold{fold_idx}_{h_tag}"
    checkpoint_path = fold_dir / "best_model.pt"
    viz_dir = fold_dir / "visualizations"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please run train_tft.py first!"
        )
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Checkpoint: {checkpoint_path}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로더
    print("\n📊 Loading data...")
    data_loader = TFTDataLoader(
        price_data_path=price_path,
        news_data_path=news_path,
        split_file=split_file,
        seq_length=config.seq_length,
        horizons=config.horizons,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Validation dates 추출
    import json
    T = data_loader.T
    T_str = np.array([str(t)[:10] for t in T])

    with open(split_file, 'r') as f:
        split_data = json.load(f)

    valid_dates_json = [
        str(d)[:10] for d in split_data['folds'][fold_idx]['val']['t_dates']
]

    valid_mask = np.isin(T_str, valid_dates_json)
    valid_dates = T[valid_mask]
    valid_dates = [str(d)[:10] for d in valid_dates]
    
    # 모델 로드
    print("\n🤖 Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Config 추출 (KeyError 방지!)
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        # Fallback: TrainConfig 사용
        model_config = {
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'attention_heads': config.attention_heads,
            'dropout': config.dropout
        }
    
    model = TemporalFusionTransformer(
        num_features=data_loader.X.shape[-1],
        num_horizons=len(config.horizons),
        hidden_dim=model_config['hidden_dim'],
        lstm_layers=model_config['num_layers'],
        attention_heads=model_config['attention_heads'],
        dropout=model_config['dropout'],
        use_variable_selection=config.use_variable_selection,
        quantiles=config.quantiles if config.quantile_loss else None
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Validation loader
    _, valid_loader = data_loader.get_fold_loaders(fold_idx)
    
    # 예측
    print("\n🔮 Predicting on validation set...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_Y in valid_loader:
            batch_X = batch_X.to(device)
            
            output = model(batch_X, return_attention=False)
            predictions = output['predictions']
            
            # Quantile median 추출
            if len(predictions.shape) == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[:, :, median_idx]
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_Y.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    num_samples = predictions.shape[0]
    valid_dates = valid_dates[-num_samples:]

    assert len(valid_dates) == predictions.shape[0], \
        f"Date mismatch: {len(valid_dates)} vs {predictions.shape[0]}"
    
    print(f"✓ Predictions shape: {predictions.shape}")
    
    # Close 변환
    print("\n💹 Converting to close prices...")
    true_close = log_return_to_close(valid_dates, targets, price_path, config.horizons)
    pred_close = log_return_to_close(valid_dates, predictions, price_path, config.horizons)
    
    # ===== Horizon별 시각화 (개별 파일!) =====
    print("\n📈 Generating visualizations...")
    
    for h_idx, horizon in enumerate(config.horizons):
        save_path = viz_dir / f"validation_h{horizon}.png"

        plot_single_horizon(
            valid_dates,
            targets[:, h_idx],
            predictions[:, h_idx],
            true_close[:, h_idx],
            pred_close[:, h_idx],
            horizon,
            save_path
        )
        
        print(f"  ✓ Saved: validation_h{horizon}.png")
    
    # CSV 저장
    print("\n💾 Saving predictions...")
    df = pd.DataFrame({'date': valid_dates})
    for h_idx, horizon in enumerate(config.horizons):
        df[f'true_log_h{horizon}'] = targets[:, h_idx]
        df[f'pred_log_h{horizon}'] = predictions[:, h_idx]
        df[f'true_close_h{horizon}'] = true_close[:, h_idx]
        df[f'pred_close_h{horizon}'] = pred_close[:, h_idx]
    
    csv_path = fold_dir / "validation_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    print(f"\n{'='*60}")
    print("✅ Visualization completed!")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  📁 Visualizations: {viz_dir}/")
    print(f"      - validation_h1.png")
    print(f"      - validation_h5.png")
    print(f"      - validation_h10.png")
    print(f"      - validation_h20.png")
    print(f"  📁 CSV: {csv_path}")
    print()


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)
