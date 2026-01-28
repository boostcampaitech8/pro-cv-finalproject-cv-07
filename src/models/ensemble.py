import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


class FoldEnsemble:
    """
    여러 fold의 예측을 앙상블하는 클래스
    """
    
    def __init__(
        self,
        fold_results: Dict,
        method: str = 'average',
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            fold_results: train_multiple_folds의 결과
            method: 'average', 'weighted', 'best'
            weights: weighted 방식일 때 사용할 가중치
        """
        self.fold_results = fold_results
        self.method = method
        self.weights = weights
        
        self.fold_indices = sorted([
            int(k.split('_')[1]) for k in fold_results.keys()
        ])
        
        # Best fold 찾기
        self.best_fold_idx = self._find_best_fold()
    
    def _find_best_fold(self) -> int:
        """
        Validation loss가 가장 낮은 fold 찾기
        
        Returns:
            best fold index
        """
        best_loss = float('inf')
        best_idx = self.fold_indices[0]
        
        for fold_idx in self.fold_indices:
            key = f'fold_{fold_idx}'
            valid_loss = self.fold_results[key]['history']['best_valid_loss']
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_idx = fold_idx
        
        return best_idx
    
    def predict(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, Dict]:
        """
        앙상블 예측
        
        Args:
            data_loader: 예측할 데이터
            device: torch device
        
        Returns:
            predictions: [num_samples, num_horizons]
            metadata: 예측 관련 정보
        """
        if self.method == 'best':
            # Best fold만 사용
            return self._predict_best_fold(data_loader, device)
        
        elif self.method == 'average':
            # 모든 fold 평균
            return self._predict_average(data_loader, device)
        
        elif self.method == 'weighted':
            # 가중 평균
            return self._predict_weighted(data_loader, device)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def _predict_best_fold(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, Dict]:
        """Best fold로만 예측"""
        key = f'fold_{self.best_fold_idx}'
        trainer = self.fold_results[key]['trainer']
        
        predictions, targets = trainer.predict(data_loader)
        
        # Quantile prediction인 경우 median 추출
        if len(predictions.shape) == 3:
            median_idx = predictions.shape[2] // 2
            predictions = predictions[:, :, median_idx]
        
        metadata = {
            'method': 'best',
            'best_fold': self.best_fold_idx,
            'best_loss': self.fold_results[key]['history']['best_valid_loss']
        }
        
        return predictions, metadata
    
    def _predict_average(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, Dict]:
        """모든 fold의 평균"""
        all_predictions = []
        
        for fold_idx in self.fold_indices:
            key = f'fold_{fold_idx}'
            trainer = self.fold_results[key]['trainer']
            
            predictions, _ = trainer.predict(data_loader)
            
            # Quantile prediction인 경우 median 추출
            if len(predictions.shape) == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[:, :, median_idx]
            
            all_predictions.append(predictions)
        
        # 평균
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        metadata = {
            'method': 'average',
            'num_folds': len(self.fold_indices),
            'fold_indices': self.fold_indices
        }
        
        return ensemble_predictions, metadata
    
    def _predict_weighted(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[np.ndarray, Dict]:
        """가중 평균"""
        # 가중치 계산 (validation loss의 역수)
        if self.weights is None:
            losses = []
            for fold_idx in self.fold_indices:
                key = f'fold_{fold_idx}'
                loss = self.fold_results[key]['history']['best_valid_loss']
                losses.append(loss)
            
            # 역수 기반 가중치 (낮은 loss일수록 높은 가중치)
            inv_losses = [1.0 / (l + 1e-8) for l in losses]
            total = sum(inv_losses)
            weights = [w / total for w in inv_losses]
        else:
            weights = self.weights
        
        # 예측
        all_predictions = []
        
        for fold_idx in self.fold_indices:
            key = f'fold_{fold_idx}'
            trainer = self.fold_results[key]['trainer']
            
            predictions, _ = trainer.predict(data_loader)
            
            # Quantile prediction인 경우 median 추출
            if len(predictions.shape) == 3:
                median_idx = predictions.shape[2] // 2
                predictions = predictions[:, :, median_idx]
            
            all_predictions.append(predictions)
        
        # 가중 평균
        ensemble_predictions = np.zeros_like(all_predictions[0])
        for pred, w in zip(all_predictions, weights):
            ensemble_predictions += pred * w
        
        metadata = {
            'method': 'weighted',
            'num_folds': len(self.fold_indices),
            'fold_indices': self.fold_indices,
            'weights': weights
        }
        
        return ensemble_predictions, metadata
    
    def get_fold_summary(self) -> Dict:
        """
        각 fold의 성능 요약
        
        Returns:
            fold별 성능 정보
        """
        summary = {}
        
        for fold_idx in self.fold_indices:
            key = f'fold_{fold_idx}'
            history = self.fold_results[key]['history']
            
            summary[f'fold_{fold_idx}'] = {
                'best_epoch': history['best_epoch'],
                'best_valid_loss': history['best_valid_loss'],
                'final_train_loss': history['train_losses'][-1],
                'final_valid_loss': history['valid_losses'][-1]
            }
        
        # Best fold 표시
        summary['best_fold'] = self.best_fold_idx
        
        return summary
    
    def save_summary(self, output_path: str):
        """성능 요약을 JSON 파일로 저장"""
        summary = self.get_fold_summary()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Fold summary saved to {output_path}")


class PredictionAggregator:
    """
    여러 fold의 예측 결과를 집계하고 분석하는 클래스
    """
    
    def __init__(self, predictions_dict: Dict[int, np.ndarray]):
        """
        Args:
            predictions_dict: {fold_idx: predictions} 형태의 딕셔너리
        """
        self.predictions_dict = predictions_dict
        self.fold_indices = sorted(predictions_dict.keys())
    
    def compute_prediction_uncertainty(self) -> np.ndarray:
        """
        예측의 불확실성 계산 (fold 간 표준편차)
        
        Returns:
            uncertainty: [num_samples, num_horizons]
        """
        all_preds = [self.predictions_dict[idx] for idx in self.fold_indices]
        all_preds = np.stack(all_preds, axis=0)  # [num_folds, num_samples, num_horizons]
        
        # Fold 간 표준편차
        uncertainty = np.std(all_preds, axis=0)
        
        return uncertainty
    
    def compute_prediction_confidence_intervals(
        self,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        신뢰구간 계산
        
        Args:
            confidence: 신뢰수준 (default: 0.95)
        
        Returns:
            lower_bound: [num_samples, num_horizons]
            upper_bound: [num_samples, num_horizons]
        """
        all_preds = [self.predictions_dict[idx] for idx in self.fold_indices]
        all_preds = np.stack(all_preds, axis=0)
        
        # Percentile 계산
        alpha = (1 - confidence) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        lower_bound = np.percentile(all_preds, lower_percentile, axis=0)
        upper_bound = np.percentile(all_preds, upper_percentile, axis=0)
        
        return lower_bound, upper_bound
    
    def get_consensus_predictions(self, method: str = 'median') -> np.ndarray:
        """
        Fold 간 합의 예측
        
        Args:
            method: 'mean' or 'median'
        
        Returns:
            consensus: [num_samples, num_horizons]
        """
        all_preds = [self.predictions_dict[idx] for idx in self.fold_indices]
        all_preds = np.stack(all_preds, axis=0)
        
        if method == 'mean':
            return np.mean(all_preds, axis=0)
        elif method == 'median':
            return np.median(all_preds, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")


def select_best_fold(
    fold_results: Dict,
    metric: str = 'valid_loss'
) -> Tuple[int, Dict]:
    """
    최적 fold 선택
    
    Args:
        fold_results: train_multiple_folds의 결과
        metric: 선택 기준 ('valid_loss', 'mae_overall', 'rmse_overall')
    
    Returns:
        best_fold_idx: 최적 fold 인덱스
        best_metrics: 최적 fold의 성능 지표
    """
    fold_indices = sorted([
        int(k.split('_')[1]) for k in fold_results.keys()
    ])
    
    best_score = float('inf')
    best_idx = fold_indices[0]
    best_metrics = None
    
    for fold_idx in fold_indices:
        key = f'fold_{fold_idx}'
        history = fold_results[key]['history']
        
        if metric == 'valid_loss':
            score = history['best_valid_loss']
        else:
            # 마지막 epoch의 metrics 사용
            # (실제로는 best epoch의 metrics를 저장하도록 수정 필요)
            score = history.get(metric, float('inf'))
        
        if score < best_score:
            best_score = score
            best_idx = fold_idx
            best_metrics = {
                'best_epoch': history['best_epoch'],
                'best_valid_loss': history['best_valid_loss']
            }
    
    print(f"\nBest fold: {best_idx}")
    print(f"Best {metric}: {best_score:.4f}")
    
    return best_idx, best_metrics


if __name__ == "__main__":
    print("Ensemble module loaded successfully")
