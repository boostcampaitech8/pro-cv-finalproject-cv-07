"""
TFT Stock Price Prediction - Main Training Script

Usage:
    python train_tft.py --target_commodity corn --seq_length 20 --fold 0 1 2 3 4 5 6 7
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tyro
from tqdm import tqdm

# 현재 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.configs.train_config import TrainConfig
from src.data.dataset_tft import TFTDataLoader, build_tft_dataset
from src.models.TFT import TemporalFusionTransformer, QuantileLoss
from src.engine.trainer_tft import TFTTrainer, train_multiple_folds
from src.models.ensemble import FoldEnsemble, select_best_fold
from src.interpretation.interpretation import (
    FeatureImportanceAnalyzer,
    AttentionAnalyzer,
    InterpretationReport
)
from src.utils.set_seed import set_seed


def main(config: TrainConfig):
    """메인 함수"""
    
    # Seed 설정
    set_seed(config.seed)
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")
    
    # ===== 데이터 경로 설정 (수정됨) =====
    # 가격 데이터와 뉴스 데이터 경로
    price_file = f"preprocessing/{config.target_commodity}_feature_engineering.csv"
    news_file = "news_features.csv"
    
    price_path = os.path.join(config.data_dir, price_file)
    news_path = os.path.join(config.data_dir, news_file)
    split_file = os.path.join(config.data_dir, "rolling_fold.json")
    
    # 경로 확인
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price data not found: {price_path}")
    if not os.path.exists(news_path):
        print(f"⚠️  News data not found: {news_path}")
        print(f"   Training without news features...")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    print(f"📁 Price data: {price_path}")
    print(f"📁 News data: {news_path}")
    print(f"📁 Split file: {split_file}")
    # ====================================
    
    # 데이터 로더 생성
    print("\n" + "="*60)
    print("Loading Data...")
    print("="*60)
    
    # ===== DataLoader 호출 수정 =====
    data_loader_manager = TFTDataLoader(
        price_data_path=price_path,  # 수정!
        news_data_path=news_path,    # 수정!
        split_file=split_file,
        seq_length=config.seq_length,
        horizons=config.horizons,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    # ===============================
    
    # Feature 이름 저장
    feature_names = data_loader_manager.feature_names
    print(f"\n✓ Number of features: {len(feature_names)}")
    print(f"✓ Horizons: {config.horizons}")
    print(f"✓ Sequence length: {config.seq_length}")
    
    # 모델 학습
    print("\n" + "="*60)
    print("Training Models...")
    print("="*60)
    
    fold_results = train_multiple_folds(
        model_class=TemporalFusionTransformer,
        data_loader_manager=data_loader_manager,
        config=config,
        device=device,
        folds=config.fold
    )
    
    # Fold 성능 요약
    print("\n" + "="*60)
    print("Fold Performance Summary")
    print("="*60)
    
    ensemble = FoldEnsemble(
        fold_results=fold_results,
        method=config.ensemble_method
    )
    
    fold_summary = ensemble.get_fold_summary()
    
    for fold_key, metrics in fold_summary.items():
        if fold_key == 'best_fold':
            print(f"\n🏆 Best Fold: {metrics}")
        else:
            print(f"\n{fold_key}:")
            print(f"  Best Epoch: {metrics['best_epoch']}")
            print(f"  Best Valid Loss: {metrics['best_valid_loss']:.4f}")
            print(f"  Final Train Loss: {metrics['final_train_loss']:.4f}")
    
    # Fold summary 저장
    summary_path = os.path.join(config.output_dir, "fold_summary.json")
    ensemble.save_summary(summary_path)
    
    # 테스트 데이터 예측
    print("\n" + "="*60)
    print("Evaluating on Test Set...")
    print("="*60)
    
    test_dates, test_loader = data_loader_manager.get_test_loader()
    
    if config.ensemble:
        # 앙상블 예측
        print(f"\nUsing ensemble method: {config.ensemble_method}")
        predictions, metadata = ensemble.predict(test_loader, device)
        
        # 메타데이터 출력
        print(f"\nEnsemble metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        # Best fold만 사용
        print("\nUsing best fold for prediction")
        best_fold_idx, _ = select_best_fold(fold_results)
        
        key = f'fold_{best_fold_idx}'
        trainer = fold_results[key]['trainer']
        predictions, targets = trainer.predict(test_loader)
        
        # Quantile prediction인 경우 median 추출
        if len(predictions.shape) == 3:
            median_idx = predictions.shape[2] // 2
            predictions = predictions[:, :, median_idx]
    
    # 결과 저장
    results_path = os.path.join(config.output_dir, "predictions")
    os.makedirs(results_path, exist_ok=True)
    
    # 예측 결과를 DataFrame으로 저장
    pred_df = pd.DataFrame(
        predictions,
        columns=[f'pred_h{h}' for h in config.horizons]
    )
    pred_df['date'] = test_dates
    
    pred_file = os.path.join(
        results_path,
        f"{config.target_commodity}_predictions.csv"
    )
    pred_df.to_csv(pred_file, index=False)
    print(f"\n✓ Predictions saved to {pred_file}")
    
    # 해석 가능성 분석
    if config.compute_feature_importance or config.compute_temporal_importance:
        print("\n" + "="*60)
        print("Interpretation Analysis...")
        print("="*60)
        
        # Best fold의 interpretation data 로드
        best_fold_idx = fold_summary['best_fold']
        
        interp_path = Path(config.interpretation_dir) / f'fold_{best_fold_idx}_interpretation.npz'
        
        if interp_path.exists():
            data = np.load(interp_path, allow_pickle=True)
            
            # Variable importance
            if config.compute_feature_importance and 'variable_importance' in data:
                var_importance = data['variable_importance']
                
                if len(var_importance) > 0:
                    # 가장 최근 epoch의 데이터 사용
                    var_importance_data = var_importance[-1]
                    
                    feature_analyzer = FeatureImportanceAnalyzer(
                        variable_importance=var_importance_data,
                        feature_names=feature_names,
                        horizons=config.horizons
                    )
            
            # Attention weights
            if config.compute_temporal_importance and 'attention_weights' in data:
                attn_weights = data['attention_weights']
                
                if len(attn_weights) > 0:
                    # 가장 최근 epoch의 데이터 사용
                    attn_weights_data = attn_weights[-1]
                    
                    attention_analyzer = AttentionAnalyzer(
                        attention_weights=attn_weights_data,
                        seq_length=config.seq_length
                    )
            
            # 종합 리포트
            if config.compute_feature_importance and config.compute_temporal_importance:
                report = InterpretationReport(
                    feature_analyzer=feature_analyzer,
                    attention_analyzer=attention_analyzer,
                    horizons=config.horizons
                )
                
                # 요약 출력
                report.print_summary()
                
                # 시각화 저장
                if config.save_attention_weights:
                    viz_dir = os.path.join(
                        config.interpretation_dir,
                        f"{config.target_commodity}_visualizations"
                    )
                    report.save_plots(viz_dir)
        else:
            print(f"\n⚠️  Interpretation data not found at {interp_path}")
    
    print("\n" + "="*60)
    print("✓ Training and evaluation completed!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Checkpoints: {config.checkpoint_dir}")
    print(f"  - Predictions: {results_path}")
    if config.compute_feature_importance or config.compute_temporal_importance:
        print(f"  - Interpretations: {config.interpretation_dir}")
    print()


if __name__ == "__main__":
    # Parse config from command line
    config = tyro.cli(TrainConfig)
    
    # Run main
    main(config)
