"""
TFT Interpretation Data Viewer

interpretation.npz 파일의 내용을 확인하고 시각화하는 도구

Usage:
    python view_interpretation.py --interp_file fold_0_interpretation.npz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_interpretation_file(file_path: str):
    """
    interpretation.npz 파일 로드
    
    Args:
        file_path: .npz 파일 경로
    
    Returns:
        dict with variable_importance and attention_weights
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    print(f"\n{'='*60}")
    print(f"Interpretation File: {file_path}")
    print(f"{'='*60}")
    
    # 파일 내용 확인
    print(f"\n📦 Contents:")
    for key in data.files:
        print(f"  - {key}")
    
    return data


def analyze_variable_importance(data: np.lib.npyio.NpzFile):
    """
    Variable importance 분석
    
    Args:
        data: loaded .npz file
    """
    if 'variable_importance' not in data:
        print("\n⚠️  No variable_importance in this file")
        return None
    
    var_importance = data['variable_importance']
    
    print(f"\n{'='*60}")
    print("📊 Variable Importance Analysis")
    print(f"{'='*60}")
    
    print(f"\nData structure:")
    print(f"  - Type: {type(var_importance)}")
    print(f"  - Shape: {len(var_importance)} epochs saved")
    
    if len(var_importance) > 0:
        # 마지막 epoch 분석
        last_epoch = var_importance[-1]
        print(f"\nLast epoch data:")
        print(f"  - Shape: {last_epoch.shape}")
        print(f"    [num_samples={last_epoch.shape[0]}, " \
              f"seq_length={last_epoch.shape[1] if len(last_epoch.shape) > 1 else 'N/A'}, " \
              f"num_features={last_epoch.shape[2] if len(last_epoch.shape) > 2 else 'N/A'}]")
        
        # 평균 중요도
        if len(last_epoch.shape) == 3:
            avg_importance = last_epoch.mean(axis=(0, 1))
            print(f"\n📈 Average importance across all samples & timesteps:")
            print(f"  Min: {avg_importance.min():.6f}")
            print(f"  Max: {avg_importance.max():.6f}")
            print(f"  Mean: {avg_importance.mean():.6f}")
            print(f"  Std: {avg_importance.std():.6f}")
            
            # Top 10 features
            top_indices = np.argsort(avg_importance)[-10:][::-1]
            print(f"\n🏆 Top 10 Most Important Features (indices):")
            for i, idx in enumerate(top_indices, 1):
                print(f"  {i}. Feature {idx}: {avg_importance[idx]:.6f}")
        
        return var_importance
    else:
        print("\n⚠️  No data in variable_importance")
        return None


def analyze_attention_weights(data: np.lib.npyio.NpzFile):
    """
    Attention weights 분석
    
    Args:
        data: loaded .npz file
    """
    if 'attention_weights' not in data:
        print("\n⚠️  No attention_weights in this file")
        return None
    
    attn_weights = data['attention_weights']
    
    print(f"\n{'='*60}")
    print("🔍 Attention Weights Analysis")
    print(f"{'='*60}")
    
    print(f"\nData structure:")
    print(f"  - Type: {type(attn_weights)}")
    print(f"  - Shape: {len(attn_weights)} epochs saved")
    
    if len(attn_weights) > 0:
        # 마지막 epoch 분석
        last_epoch = attn_weights[-1]
        print(f"\nLast epoch data:")
        print(f"  - Shape: {last_epoch.shape}")
        if len(last_epoch.shape) == 4:
            print(f"    [num_samples={last_epoch.shape[0]}, " \
                  f"num_heads={last_epoch.shape[1]}, " \
                  f"seq_length={last_epoch.shape[2]}, " \
                  f"seq_length={last_epoch.shape[3]}]")
        
        # 평균 attention
        if len(last_epoch.shape) == 4:
            avg_attention = last_epoch.mean(axis=(0, 1))  # [seq_length, seq_length]
            print(f"\n📈 Average attention (across samples & heads):")
            print(f"  Shape: {avg_attention.shape}")
            print(f"  Min: {avg_attention.min():.6f}")
            print(f"  Max: {avg_attention.max():.6f}")
            print(f"  Sum per row: {avg_attention.sum(axis=1).mean():.6f} (should be ~1.0)")
            
            # 각 timestep의 중요도
            temporal_importance = avg_attention.mean(axis=0)
            print(f"\n⏰ Temporal Importance (average attention received):")
            for t in range(len(temporal_importance)):
                print(f"  t-{len(temporal_importance)-t-1}: {temporal_importance[t]:.6f}")
        
        return attn_weights
    else:
        print("\n⚠️  No data in attention_weights")
        return None


def visualize_attention_heatmap(
    attn_weights: np.ndarray,
    save_path: str = "attention_heatmap.png"
):
    """
    Attention weights heatmap 시각화
    
    Args:
        attn_weights: attention weights array (epochs saved)
        save_path: 저장 경로
    """
    if attn_weights is None or len(attn_weights) == 0:
        print("\n⚠️  No attention weights to visualize")
        return
    
    # 마지막 epoch 사용
    last_epoch = attn_weights[-1]
    
    if len(last_epoch.shape) != 4:
        print(f"\n⚠️  Unexpected shape: {last_epoch.shape}")
        return
    
    # 평균 attention
    avg_attention = last_epoch.mean(axis=(0, 1))
    seq_length = avg_attention.shape[0]
    
    # 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_attention,
        xticklabels=[f't-{seq_length-i-1}' for i in range(seq_length)],
        yticklabels=[f't-{seq_length-i-1}' for i in range(seq_length)],
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'},
        annot=False,
        fmt='.3f'
    )
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Average Attention Weights\n(across all samples & heads)')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Attention heatmap saved: {save_path}")


def visualize_variable_importance(
    var_importance: np.ndarray,
    feature_names: list = None,
    top_k: int = 20,
    save_path: str = "variable_importance.png"
):
    """
    Variable importance 시각화
    
    Args:
        var_importance: variable importance array (epochs saved)
        feature_names: feature 이름 리스트 (optional)
        top_k: 상위 몇 개
        save_path: 저장 경로
    """
    if var_importance is None or len(var_importance) == 0:
        print("\n⚠️  No variable importance to visualize")
        return
    
    # 마지막 epoch 사용
    last_epoch = var_importance[-1]
    
    if len(last_epoch.shape) != 3:
        print(f"\n⚠️  Unexpected shape: {last_epoch.shape}")
        return
    
    # 평균 importance
    avg_importance = last_epoch.mean(axis=(0, 1))
    
    # Top k 추출
    top_indices = np.argsort(avg_importance)[-top_k:][::-1]
    top_values = avg_importance[top_indices]
    
    # Feature 이름
    if feature_names is not None:
        top_names = [feature_names[i] if i < len(feature_names) else f'Feature {i}' 
                     for i in top_indices]
    else:
        top_names = [f'Feature {i}' for i in top_indices]
    
    # 시각화
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_values)), top_values, color='steelblue')
    plt.yticks(range(len(top_values)), top_names)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_k} Variable Importance')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Variable importance plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='View TFT interpretation.npz file')
    parser.add_argument('--interp_file', type=str, required=True,
                       help='Path to fold_X_interpretation.npz')
    parser.add_argument('--output_dir', type=str, default='./interpretation_analysis',
                       help='Output directory for plots')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of top features to show')
    
    args = parser.parse_args()
    
    # 파일 로드
    data = load_interpretation_file(args.interp_file)
    
    # Variable importance 분석
    var_importance = analyze_variable_importance(data)
    
    # Attention weights 분석
    attn_weights = analyze_attention_weights(data)
    
    # 시각화
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if attn_weights is not None:
        visualize_attention_heatmap(
            attn_weights,
            save_path=str(Path(args.output_dir) / 'attention_heatmap.png')
        )
    
    if var_importance is not None:
        visualize_variable_importance(
            var_importance,
            top_k=args.top_k,
            save_path=str(Path(args.output_dir) / f'top{args.top_k}_variable_importance.png')
        )
    
    print(f"\n{'='*60}")
    print("✅ Analysis completed!")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
