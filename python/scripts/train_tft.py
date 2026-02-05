"""
TFT Training Script - Final Version

폴더 구조:
checkpoints/TFT_corn_fold0/
  ├── best_model.pt
  ├── visualizations/
  │   ├── loss_curve.png
  │   ├── validation_h1.png
  │   ├── validation_h5.png
  │   ├── validation_h10.png
  │   └── validation_h20.png
  └── interpretations/
      ├── feature_importance.png
      ├── attention_heatmap.png
      └── interpretation_data.npz
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tyro

sys.path.append(str(Path(__file__).parent.parent))

from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset_tft import TFTDataLoader
from src.models.TFT import TemporalFusionTransformer, QuantileLoss
from src.engine.trainer_tft import train
from src.utils.visualization import save_loss_curve


def main(config: TrainConfig):
    set_seed(config.seed)

    h_tag = "h" + "-".join(map(str, config.horizons))
    print(f"[DEBUG] horizons = {config.horizons} -> h_tag = {h_tag}")

    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    
    all_summaries = {}
    
    # ===== Loop: commodities =====
    for commodity in [config.target_commodity]:
        print(f"\n{'='*60}")
        print(f"Training: {commodity.upper()}")
        print(f"{'='*60}\n")
        
        # Data paths
        price_file = f"preprocessing/{commodity}_feature_engineering.csv"
        news_file = "news_features.csv"
        
        price_path = os.path.join(config.data_dir, price_file)
        news_path = os.path.join(config.data_dir, news_file)
        split_file = os.path.join(config.data_dir, "rolling_fold.json")
        
        if not os.path.exists(price_path):
            print(f"{price_path} 파일 존재하지 않음")
            return
        
        # Load data
        print("Loading data...")
        data_loader = TFTDataLoader(
            price_data_path=price_path,
            news_data_path=news_path,
            split_file=split_file,
            seq_length=config.seq_length,
            horizons=config.horizons,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        print(f"✓ Features: {data_loader.X.shape[-1]}")
        print(f"✓ Horizons: {config.horizons}")
        
        commodity_summary = {}
        
        # ===== Loop: folds =====
        for fold in config.fold:
            print(f"\n{'='*60}")
            print(f"Fold {fold}")
            print(f"{'='*60}\n")
            
            # ===== 폴더 구조 생성 =====
            fold_dir = Path(config.checkpoint_dir) / f"TFT_{commodity}_fold{fold}_{h_tag}"
            viz_dir = fold_dir / "visualizations"
            interp_dir = fold_dir / "interpretations"
            
            fold_dir.mkdir(parents=True, exist_ok=True)
            viz_dir.mkdir(exist_ok=True)
            interp_dir.mkdir(exist_ok=True)
            
            # Get loaders
            train_loader, valid_loader = data_loader.get_fold_loaders(fold)
            
            # Model
            model = TemporalFusionTransformer(
                num_features=data_loader.X.shape[-1],
                num_horizons=len(config.horizons),
                hidden_dim=config.hidden_dim,
                lstm_layers=config.num_layers,
                attention_heads=config.attention_heads,
                dropout=config.dropout,
                use_variable_selection=config.use_variable_selection,
                quantiles=config.quantiles if config.quantile_loss else None
            )
            model = model.to(device)
            
            # Optimizer & Criterion
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
            
            if config.quantile_loss:
                criterion = QuantileLoss(quantiles=config.quantiles)
            else:
                criterion = nn.MSELoss()
            
            # Train!
            model, train_hist, valid_hist, best_metrics, best_epoch, best_val_loss = train(
                model, train_loader, valid_loader,
                criterion, optimizer, device,
                num_epochs=config.epochs,
                patience=config.early_stopping_patience
            )
            
            # ===== Save checkpoint (개선된 구조) =====
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "train_hist": train_hist,
                "valid_hist": valid_hist,
                "best_metrics": best_metrics,
                "config": {
                    "hidden_dim": config.hidden_dim,
                    "num_layers": config.num_layers,
                    "attention_heads": config.attention_heads,
                    "dropout": config.dropout,
                    "horizons": config.horizons,
                    "seq_length": config.seq_length
                }
            }
            
            checkpoint_path = fold_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"\n✓ Checkpoint saved: {checkpoint_path}")
            
            # ===== Save loss curve =====
            loss_curve_path = viz_dir / "loss_curve.png"
            save_loss_curve(
                train_hist, valid_hist,
                str(viz_dir),
                "loss_curve.png"
            )
            print(f"✓ Loss curve saved: {loss_curve_path}")
            
            # ===== Interpretations (Variable importance & Attention) =====
            if config.compute_feature_importance or config.compute_temporal_importance:
                print("\n📊 Computing interpretations...")
                
                # Validation 데이터로 interpretation 계산
                model.eval()
                all_var_importance = []
                all_attn_weights = []
                
                with torch.no_grad():
                    for x_val, y_val in valid_loader:
                        x_val = x_val.to(device)
                        
                        outputs = model(x_val, return_attention=True)
                        
                        vi = outputs.get('variable_importance', None)

                        if vi is not None and config.compute_feature_importance:
                            all_var_importance.append(vi.detach().cpu().numpy())
                        
                        if 'attention_weights' in outputs and config.compute_temporal_importance:
                            all_attn_weights.append(outputs['attention_weights'].cpu().numpy())
                
                # Save interpretation data
                interp_data = {}
                if all_var_importance:
                    var_importance = np.concatenate(all_var_importance, axis=0).mean(axis=0)
                    interp_data['variable_importance'] = var_importance

                if all_attn_weights and config.compute_temporal_importance:
                    # all_attn_weights: List of [batch, heads, seq, seq]
                    attn = np.concatenate(all_attn_weights, axis=0)  # [N, H, T, T]
                    avg_attn = attn.mean(axis=(0, 1))  # [T, T]
                    interp_data['attention_weights'] = avg_attn
                
                if interp_data:
                    interp_path = interp_dir / "interpretation_data.npz"
                    np.savez(interp_path, **interp_data)
                    print(f"✓ Interpretations saved: {interp_path}")
                    
                    # Visualization (간단)
                    try:
                        import matplotlib.pyplot as plt
                        
                        # Variable importance
                        if 'variable_importance' in interp_data:
                            plt.figure(figsize=(10, 6))
                            importance = interp_data['variable_importance']
                            top_k = min(20, len(importance))
                            top_indices = np.argsort(importance)[-top_k:]
                            
                            plt.barh(range(top_k), importance[top_indices])
                            plt.xlabel('Importance')
                            plt.title('Top Feature Importance')
                            plt.tight_layout()
                            plt.savefig(interp_dir / "feature_importance.png", dpi=150)
                            plt.close()
                            print(f"✓ Feature importance plot saved")
                        
                        # Attention heatmap
                        if 'attention_weights' in interp_data:
                            plt.figure(figsize=(10, 6))
                            attn = interp_data['attention_weights']
                            plt.imshow(attn.T, aspect='auto', cmap='viridis')
                            plt.colorbar()
                            plt.xlabel('Time Step')
                            plt.ylabel('Horizon')
                            plt.title('Temporal Attention Weights')
                            plt.tight_layout()
                            plt.savefig(interp_dir / "attention_heatmap.png", dpi=150)
                            plt.close()
                            print(f"✓ Attention heatmap saved")
                    except Exception as e:
                        print(f"⚠️  Visualization failed: {e}")
            
            # Fold summary
            if best_metrics:
                fold_summary = {
                    'best_epoch': int(best_epoch),
                    'best_valid_loss': float(best_val_loss),
                    'final_train_loss': float(train_hist[-1]),
                    'mae_overall': float(best_metrics['mae_overall']),
                    'rmse_overall': float(best_metrics['rmse_overall']),
                    'da_overall': float(best_metrics['da_overall']),
                    'r2_overall': float(best_metrics['r2_overall']),
                }
                
                for h in range(len(config.horizons)):
                    fold_summary[f'mae_h{h}'] = float(best_metrics[f'mae_h{h}'])
                    fold_summary[f'rmse_h{h}'] = float(best_metrics[f'rmse_h{h}'])
                    fold_summary[f'da_h{h}'] = float(best_metrics[f'da_h{h}'])
                    fold_summary[f'r2_h{h}'] = float(best_metrics[f'r2_h{h}'])
                
                commodity_summary[f'fold_{fold}'] = fold_summary
        
        if commodity_summary:
            best_fold = min(commodity_summary.items(), 
                          key=lambda x: x[1]['best_valid_loss'])[0]
            commodity_summary['best_fold'] = best_fold
        
        all_summaries[commodity] = commodity_summary
    
    # Save summary JSON
    summary_path = os.path.join(config.output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Training completed!")
    print(f"{'='*60}")
    print(f"\n📁 Outputs:")
    print(f"   Checkpoint: {fold_dir}/best_model.pt")
    print(f"   Visualizations: {viz_dir}/")
    print(f"   Interpretations: {interp_dir}/")
    print(f"   Summary: {summary_path}\n")


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)
