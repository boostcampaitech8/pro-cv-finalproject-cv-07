"""
TFT Test Script - Final Version

저장 위치: checkpoints/TFT_corn_fold0/visualizations/
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import tyro

sys.path.append(str(Path(__file__).parent.parent))

from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset_tft import TFTDataLoader
from src.models.TFT import TemporalFusionTransformer
from src.data.postprocessing import convert_close


def compute_metrics_per_horizon(predictions, targets, horizons):
    """Horizon별 metrics 계산"""
    num_horizons = targets.shape[1]
    
    for h_idx, horizon in enumerate(horizons):
        pred_h = predictions[:, h_idx]
        true_h = targets[:, h_idx]
        
        mae = np.mean(np.abs(pred_h - true_h))
        rmse = np.sqrt(np.mean((pred_h - true_h) ** 2))
        
        epsilon = 1e-8
        mape = np.mean(np.abs((true_h - pred_h) / (np.abs(true_h) + epsilon))) * 100
        
        ss_res = np.sum((true_h - pred_h) ** 2)
        ss_tot = np.sum((true_h - np.mean(true_h)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))
        
        da = np.mean((pred_h > 0) == (true_h > 0)) * 100
        
        # Pretty print
        print(f"\nHorizon {horizon} Metrics:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.4f}")
        print(f"  DA: {da:.2f}%")


def plot_single_horizon_test(
    dates, true_log, pred_log, true_close, pred_close,
    horizon, save_path
):
    """단일 horizon 테스트 시각화"""
    dates_dt = [datetime.strptime(str(d)[:10], '%Y-%m-%d') for d in dates]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Log Return
    ax = axes[0, 0]
    ax.plot(dates_dt, true_log, label='True', color='blue', linewidth=1.5)
    ax.plot(dates_dt, pred_log, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Log Return', fontsize=11)
    ax.set_title(f'Test - H{horizon} - Log Return', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Log Return Error
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
    
    # Close Price
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
    
    # Close Price Error
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


def test(model, test_loader, device):
    """Test (predict)"""
    model.eval()
    
    preds = []
    trues = []
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            
            outputs = model(x_test, return_attention=False)
            predictions = outputs['predictions']
            
            if len(predictions.shape) == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[:, :, median_idx]
            
            preds.append(predictions.cpu())
            trues.append(y_test)
    
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    
    return preds, trues


def main(config: TrainConfig):
    set_seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    
    for commodity in [config.target_commodity]:
        print(f"\n{'='*60}")
        print(f"Testing: {commodity.upper()}")
        print(f"{'='*60}\n")
        
        price_file = f"preprocessing/{commodity}_feature_engineering.csv"
        news_file = "news_features.csv"
        
        price_path = os.path.join(config.data_dir, price_file)
        news_path = os.path.join(config.data_dir, news_file)
        split_file = os.path.join(config.data_dir, "rolling_fold.json")
        
        if not os.path.exists(price_path):
            print(f"{price_path} 파일 존재하지 않음")
            return
        
        data_loader = TFTDataLoader(
            price_data_path=price_path,
            news_data_path=news_path,
            split_file=split_file,
            seq_length=config.seq_length,
            horizons=config.horizons,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        data = pd.read_csv(price_path)
        test_dates, test_loader = data_loader.get_test_loader()
        
        all_preds = []
        all_trues = []
        
        for fold in config.fold:
            # ===== 폴더 구조 맞춤 =====
            h_tag = "h" + "-".join(map(str, config.horizons))
            fold_dir = Path(config.checkpoint_dir) / f"TFT_{commodity}_fold{fold}_{h_tag}"
            checkpoint_path = fold_dir / "best_model.pt"
            viz_dir = fold_dir / "visualizations"
            
            if not checkpoint_path.exists():
                print(f"{checkpoint_path} 파일 존재하지 않음")
                return
            
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'config' in checkpoint:
                model_config = checkpoint['config']
            else:
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
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            
            preds, trues = test(model, test_loader, device)
            
            all_preds.append(preds)
            all_trues.append(trues)
        
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        
        final_pred = all_preds[0] if all_preds.shape[0] == 1 else all_preds.mean(axis=0)
        final_true = all_trues[0]
        
        # Metrics
        print("\n" + "="*60)
        print("Test Metrics")
        print("="*60)
        compute_metrics_per_horizon(final_pred, final_true, config.horizons)
        
        # Close 변환
        true_close, pred_close = convert_close(data, test_dates, final_true, final_pred, config.horizons)
        
        # ===== Horizon별 시각화 =====
        print("\n📈 Generating visualizations...")
        
        for h_idx, horizon in enumerate(config.horizons):
            save_path = viz_dir / f"test_h{horizon}.png"
            
            plot_single_horizon_test(
                test_dates,
                final_true[:, h_idx],
                final_pred[:, h_idx],
                [tc[h_idx] for tc in true_close],
                [pc[h_idx] for pc in pred_close],
                horizon,
                save_path
            )
            
            print(f"  ✓ Saved: test_h{horizon}.png")
        
        # CSV 저장
        pred_close_array = np.array(pred_close).T
        close_columns = [f"close_{h}" for h in config.horizons]
        
        close_df = pd.DataFrame(pred_close_array, columns=close_columns)
        close_df.insert(0, "time", test_dates[:len(pred_close_array)])
        
        csv_path = fold_dir / "test_predictions.csv"
        close_df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Predictions saved: {csv_path}")
        print(f"✓ Visualizations saved: {viz_dir}/")
    
    print(f"\n{'='*60}")
    print("✅ Testing completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)
