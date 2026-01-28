import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path


class TFTTrainer:
    """
    TFT 모델 학습을 위한 Trainer
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: dict,
        fold_index: int = 0
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.fold_index = fold_index
        
        # History
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float('inf')
        self.best_epoch = 0
        
        # Early stopping
        self.patience = config.get('early_stopping_patience', 20)
        self.patience_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Interpretation storage
        self.variable_importance_history = []
        self.attention_weights_history = []
    
    def train_epoch(self) -> float:
        """
        한 epoch 학습
        
        Returns:
            평균 train loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Fold {self.fold_index} - Training")
        
        for batch_X, batch_Y in pbar:
            # Move to device
            batch_X = batch_X.to(self.device)  # [batch_size, seq_length, num_features]
            batch_Y = batch_Y.to(self.device)  # [batch_size, num_horizons]
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch_X)
            
            predictions = output['predictions']
            
            # Loss 계산
            loss = self.criterion(predictions, batch_Y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """
        Validation
        
        Returns:
            avg_loss: 평균 validation loss
            metrics: 추가 평가 지표들
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_var_importance = []
        all_attn_weights = []
        
        with torch.no_grad():
            for batch_X, batch_Y in self.valid_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                
                # Forward pass
                output = self.model(batch_X, return_attention=True)
                predictions = output['predictions']
                
                # Loss
                loss = self.criterion(predictions, batch_Y)
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_Y.cpu().numpy())
                
                # Store interpretation data
                if 'variable_importance' in output and output['variable_importance'] is not None:
                    all_var_importance.append(output['variable_importance'].cpu().numpy())
                
                if 'attention_weights' in output and output['attention_weights'] is not None:
                    all_attn_weights.append(output['attention_weights'].cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        # Store interpretation data
        if len(all_var_importance) > 0:
            self.variable_importance_history.append(np.concatenate(all_var_importance, axis=0))
        
        if len(all_attn_weights) > 0:
            self.attention_weights_history.append(np.concatenate(all_attn_weights, axis=0))
        
        return avg_loss, metrics
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        평가 지표 계산
        
        Args:
            predictions: [num_samples, num_horizons] or [num_samples, num_horizons, num_quantiles]
            targets: [num_samples, num_horizons]
        
        Returns:
            metrics dict
        """
        metrics = {}
        
        # Quantile prediction인 경우 median (0.5 quantile) 사용
        if len(predictions.shape) == 3:
            # Find median quantile index
            median_idx = predictions.shape[2] // 2
            pred_median = predictions[:, :, median_idx]
        else:
            pred_median = predictions
        
        # Horizon별 MAE, RMSE
        num_horizons = targets.shape[1]
        
        for h in range(num_horizons):
            pred_h = pred_median[:, h]
            target_h = targets[:, h]
            
            mae = np.mean(np.abs(pred_h - target_h))
            rmse = np.sqrt(np.mean((pred_h - target_h) ** 2))
            
            metrics[f'mae_h{h}'] = mae
            metrics[f'rmse_h{h}'] = rmse
        
        # Overall MAE, RMSE
        mae_overall = np.mean(np.abs(pred_median - targets))
        rmse_overall = np.sqrt(np.mean((pred_median - targets) ** 2))
        
        metrics['mae_overall'] = mae_overall
        metrics['rmse_overall'] = rmse_overall
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict:
        """
        전체 학습 프로세스
        
        Args:
            num_epochs: 학습 epoch 수
        
        Returns:
            학습 히스토리
        """
        print(f"\n{'='*50}")
        print(f"Training Fold {self.fold_index}")
        print(f"{'='*50}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            valid_loss, metrics = self.validate()
            self.valid_losses.append(valid_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Valid Loss: {valid_loss:.4f}")
            print(f"Valid MAE: {metrics['mae_overall']:.4f}")
            print(f"Valid RMSE: {metrics['rmse_overall']:.4f}")
            
            # Check for improvement
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, metrics, is_best=True)
                print(f"✓ New best model saved! (epoch {epoch + 1})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best epoch: {self.best_epoch + 1}, Best valid loss: {self.best_valid_loss:.4f}")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, metrics, is_best=False)
        
        # Load best model
        self.load_best_model()
        
        # Return history
        history = {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'best_epoch': self.best_epoch,
            'best_valid_loss': self.best_valid_loss,
            'variable_importance_history': self.variable_importance_history,
            'attention_weights_history': self.attention_weights_history
        }
        
        return history
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'best_valid_loss': self.best_valid_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / f'fold_{self.fold_index}_best.pt'
        else:
            path = self.checkpoint_dir / f'fold_{self.fold_index}_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
    
    def load_best_model(self):
        """Best 모델 로드"""
        path = self.checkpoint_dir / f'fold_{self.fold_index}_best.pt'
        
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
        else:
            print(f"\nWarning: Best model checkpoint not found at {path}")
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행
        
        Args:
            data_loader: 예측할 데이터
        
        Returns:
            predictions: [num_samples, num_horizons] or [num_samples, num_horizons, num_quantiles]
            targets: [num_samples, num_horizons]
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_Y in tqdm(data_loader, desc="Predicting"):
                batch_X = batch_X.to(self.device)
                
                output = self.model(batch_X, return_attention=False)
                predictions = output['predictions']
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_Y.numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, targets
    
    def get_interpretation_data(self) -> Dict:
        """
        해석 가능성 데이터 반환
        
        Returns:
            dict with variable importance and attention weights
        """
        return {
            'variable_importance': self.variable_importance_history,
            'attention_weights': self.attention_weights_history
        }


def train_multiple_folds(
    model_class,
    data_loader_manager,
    config,
    device,
    folds: List[int]
) -> Dict:
    """
    여러 fold를 학습
    
    Args:
        model_class: 모델 클래스
        data_loader_manager: TFTDataLoader 인스턴스
        config: 설정
        device: torch device
        folds: 학습할 fold 리스트
    
    Returns:
        각 fold의 학습 결과
    """
    results = {}
    
    for fold_idx in folds:
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_idx}")
        print(f"{'='*60}")
        
        # Get data loaders for this fold
        train_loader, valid_loader = data_loader_manager.get_fold_loaders(fold_idx)
        
        # Create model
        model = model_class(
            num_features=data_loader_manager.X.shape[-1],
            num_horizons=len(config.horizons),
            hidden_dim=config.hidden_dim,
            lstm_layers=config.lstm_layers,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
            use_variable_selection=config.use_variable_selection,
            quantiles=config.quantiles if config.quantile_loss else None
        )
        model = model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Create criterion
        if config.quantile_loss:
            from TFT import QuantileLoss
            criterion = QuantileLoss(quantiles=config.quantiles)
        else:
            criterion = nn.MSELoss()
        
        # Create trainer
        trainer = TFTTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config.__dict__,
            fold_index=fold_idx
        )
        
        # Train
        history = trainer.train(num_epochs=config.epochs)
        
        # Store results
        results[f'fold_{fold_idx}'] = {
            'trainer': trainer,
            'model': model,
            'history': history
        }
        
        # Save interpretation data
        interp_data = trainer.get_interpretation_data()
        interp_path = Path(config.interpretation_dir) / f'fold_{fold_idx}_interpretation.npz'
        interp_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            interp_path,
            variable_importance=interp_data['variable_importance'],
            attention_weights=interp_data['attention_weights']
        )
        
        print(f"\nFold {fold_idx} completed!")
        print(f"Best valid loss: {history['best_valid_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    # 테스트 코드
    print("TFT Trainer module loaded successfully")
