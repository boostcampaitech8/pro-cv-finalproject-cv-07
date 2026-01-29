"""
TFT Trainer - Simple Version with Metrics
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm


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


def validation(model, valid_loader, criterion, device, compute_detailed_metrics=False):
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
            
            # Quantile median 추출
            if len(predictions.shape) == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[:, :, median_idx]
            
            loss = criterion(predictions, y_val)
            avg_valid_loss += loss.item()
            
            if compute_detailed_metrics:
                all_preds.append(predictions.cpu().numpy())
                all_trues.append(y_val.cpu().numpy())
    
    avg_valid_loss /= len(valid_loader)
    
    if compute_detailed_metrics:
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        metrics = compute_metrics(all_preds, all_trues)
        return avg_valid_loss, metrics
    
    return avg_valid_loss, None


def train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=100, patience=20):
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
            
            # Quantile median
            if len(predictions.shape) == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[:, :, median_idx]
            
            loss = criterion(predictions, y_train)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            avg_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss /= len(train_loader)
        train_hist[epoch] = avg_train_loss
        
        # ===== Validation =====
        # 마지막 epoch에서만 detailed metrics 계산
        compute_detailed = (epoch == num_epochs - 1) or (patience_counter >= patience - 1)
        avg_valid_loss, metrics = validation(model, valid_loader, criterion, device, compute_detailed)
        valid_hist[epoch] = avg_valid_loss
        
        # ===== Best model =====
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Detailed metrics for best model
            if metrics is None:
                _, metrics = validation(model, valid_loader, criterion, device, compute_detailed_metrics=True)
            
            best_metrics = metrics
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
        print(f'   Overall - MAE: {best_metrics["mae_overall"]:.6f}, RMSE: {best_metrics["rmse_overall"]:.6f}, DA: {best_metrics["da_overall"]:.2f}%, R²: {best_metrics["r2_overall"]:.4f}')
    
    return model.eval(), train_hist[:epoch+1], valid_hist[:epoch+1], best_metrics
