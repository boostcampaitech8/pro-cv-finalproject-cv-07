"""
TFT Trainer - Simple Version with Metrics
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm

def pick_point_pred(preds, quantiles=None):
    # preds: [B,H] or [B,H,Q]
    if preds.ndim == 2:
        return preds
    Q = preds.shape[2]
    if quantiles is not None and 0.5 in quantiles:
        q_idx = quantiles.index(0.5)
    else:
        q_idx = Q // 2
    return preds[:, :, q_idx]

def compute_metrics(predictions, targets):
    """
    Compute metrics for each horizon
    
    Returns:
        dict: {'mae_h0': ..., 'rmse_h0': ..., 'da_h0': ..., ...}
    """
    metrics = {}
    
    num_horizons = targets.shape[1]
    
    for h in range(num_horizons):
        pred_h = predictions[:, h]
        true_h = targets[:, h]
        
        # MAE, RMSE
        mae = np.mean(np.abs(pred_h - true_h))
        rmse = np.sqrt(np.mean((pred_h - true_h) ** 2))
        
        # DA (Direction Accuracy)
        da = np.mean((pred_h > 0) == (true_h > 0)) * 100
        
        # R²
        ss_res = np.sum((true_h - pred_h) ** 2)
        ss_tot = np.sum((true_h - np.mean(true_h)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics[f'mae_h{h}'] = mae
        metrics[f'rmse_h{h}'] = rmse
        metrics[f'da_h{h}'] = da
        metrics[f'r2_h{h}'] = r2
    
    # Overall
    metrics['mae_overall'] = np.mean(np.abs(predictions - targets))
    metrics['rmse_overall'] = np.sqrt(np.mean((predictions - targets) ** 2))
    metrics['da_overall'] = np.mean((predictions > 0) == (targets > 0)) * 100
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics['r2_overall'] = 1 - (ss_res / (ss_tot + 1e-8))
    
    return metrics


def validation(model, valid_loader, criterion, device, compute_detailed_metrics=False, y_scaler=None):
    """
    Validation
    
    Returns:
        avg_valid_loss: float
        metrics: dict (if compute_detailed_metrics=True)
    """
    model.eval()
    avg_valid_loss = 0
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            outputs = model(x_val, return_attention=False)
            predictions = outputs['predictions']

            loss = criterion(predictions, y_val)

            predictions_point = pick_point_pred(predictions, getattr(criterion, "quantiles", None))
            
            avg_valid_loss += loss.item()
            
            if compute_detailed_metrics:
                all_preds.append(predictions_point.cpu().numpy())
                all_trues.append(y_val.cpu().numpy())
    
    avg_valid_loss /= len(valid_loader)
    
    if compute_detailed_metrics:
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        if y_scaler is not None:
            all_preds = y_scaler.inverse_transform(all_preds)
            all_trues = y_scaler.inverse_transform(all_trues)

        metrics = compute_metrics(all_preds, all_trues)
        return avg_valid_loss, metrics
    
    return avg_valid_loss, None


def train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=300, patience=30, horizons=None, y_scaler=None):
    """
    TFT Training (Simple version with metrics)
    """
    train_hist = np.zeros(num_epochs)
    valid_hist = np.zeros(num_epochs)
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # ===== Train =====
        model.train()
        avg_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x_train, y_train in pbar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            # Forward
            outputs = model(x_train, return_attention=False)
            predictions = outputs['predictions']

            loss = criterion(predictions, y_train)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            avg_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss /= len(train_loader)
        train_hist[epoch] = avg_train_loss
        
        # ===== Validation =====
        avg_valid_loss, _ = validation(
            model, valid_loader, criterion, device,
            compute_detailed_metrics=False,
            y_scaler=y_scaler
        )
        valid_hist[epoch] = avg_valid_loss

        
        # ===== Best model =====
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())

            # best model일 때만 detailed metrics 계산
            _, best_metrics = validation(
                model, valid_loader, criterion, device,
                compute_detailed_metrics=True,
                y_scaler=y_scaler
            )
            patience_counter = 0
            print(f'Epoch [{epoch+1:03d}/{num_epochs}] | Train: {avg_train_loss:.4f} | Valid: {avg_valid_loss:.4f} ✓')
        else:
            patience_counter += 1
            print(f'Epoch [{epoch+1:03d}/{num_epochs}] | Train: {avg_train_loss:.4f} | Valid: {avg_valid_loss:.4f}')

        
        # ===== Early stopping =====
        if patience_counter >= patience:
            print(f'\n⏹️  Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(best_model_wts)
    print(f'✅ Best model saved at epoch {best_epoch} (val loss = {best_val_loss:.4f})')
    
    # Best metrics 출력
    if best_metrics:
        print(f'\n📊 Best Model Validation Metrics:')
        print(
            f'   Overall - MAE: {best_metrics["mae_overall"]:.6f}, '
            f'RMSE: {best_metrics["rmse_overall"]:.6f}, '
            f'DA: {best_metrics["da_overall"]:.2f}%, '
            f'R²: {best_metrics["r2_overall"]:.4f}'
        )

        num_h = len([k for k in best_metrics.keys() if k.startswith("mae_h")])
        if horizons is None:
            horizons = list(range(num_h))

        print(f'\n   Horizon-wise:')
        for h_idx in range(num_h):
            h_name = horizons[h_idx] if h_idx < len(horizons) else h_idx
            print(
                f'   H{h_name} - '
                f'MAE: {best_metrics[f"mae_h{h_idx}"]:.6f}, '
                f'RMSE: {best_metrics[f"rmse_h{h_idx}"]:.6f}, '
                f'DA: {best_metrics[f"da_h{h_idx}"]:.2f}%, '
                f'R²: {best_metrics[f"r2_h{h_idx}"]:.4f}'
            )
    
    return model.eval(), train_hist[:epoch+1], valid_hist[:epoch+1], best_metrics, best_epoch, best_val_loss
