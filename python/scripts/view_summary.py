"""
Training Summary Viewer - LSTM Style

Usage:
    python view_summary.py --commodity corn
    python view_summary.py --commodity all
"""

import os
import json
import argparse


def print_commodity_summary(commodity, summary):
    """Commodity별 요약 출력 (LSTM 스타일)"""
    
    print(f"\n{'='*60}")
    print(f"{commodity.upper()} - Training Summary")
    print(f"{'='*60}")
    
    for fold_key, fold_data in summary.items():
        if fold_key == 'best_fold':
            continue
        
        print(f"\n{fold_key.upper()}:")
        print(f"  Best Epoch: {fold_data['best_epoch']}")
        print(f"  Best Valid Loss: {fold_data['best_valid_loss']:.6f}")
        print(f"  Final Train Loss: {fold_data['final_train_loss']:.6f}")
        print(f"\n  Overall Metrics:")
        print(f"    MAE:  {fold_data['mae_overall']:.6f}")
        print(f"    RMSE: {fold_data['rmse_overall']:.6f}")
        print(f"    DA:   {fold_data['da_overall']:.2f}%")
        print(f"    R²:   {fold_data['r2_overall']:.4f}")
        
        # Horizon별
        print(f"\n  Horizon Metrics:")
        horizons = [1, 5, 10, 20]  # default
        for h_idx, horizon in enumerate(horizons):
            if f'mae_h{h_idx}' in fold_data:
                print(f"    H{horizon:2d} - MAE: {fold_data[f'mae_h{h_idx}']:.6f}, RMSE: {fold_data[f'rmse_h{h_idx}']:.6f}, DA: {fold_data[f'da_h{h_idx}']:.2f}%, R²: {fold_data[f'r2_h{h_idx}']:.4f}")
    
    if 'best_fold' in summary:
        print(f"\n  🏆 Best Fold: {summary['best_fold']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commodity', type=str, default='all', 
                       help='Commodity name (corn/wheat/soybean) or "all"')
    parser.add_argument('--summary_file', type=str, 
                       default='/data/ephemeral/home/pro-cv-finalproject-cv-07/python/src/outputs/training_summary.json',
                       help='Path to training_summary.json')
    args = parser.parse_args()
    
    # Load summary
    if not os.path.exists(args.summary_file):
        print(f"❌ Summary file not found: {args.summary_file}")
        print("\nPlease run train_tft.py first!")
        return
    
    with open(args.summary_file, 'r') as f:
        all_summaries = json.load(f)
    
    # Print
    if args.commodity == 'all':
        for commodity, summary in all_summaries.items():
            print_commodity_summary(commodity, summary)
    else:
        if args.commodity in all_summaries:
            print_commodity_summary(args.commodity, all_summaries[args.commodity])
        else:
            print(f"❌ Commodity '{args.commodity}' not found in summary")
            print(f"Available: {list(all_summaries.keys())}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
