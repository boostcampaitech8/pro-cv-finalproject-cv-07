import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class FeatureImportanceAnalyzer:
    """
    TFT 모델의 변수 중요도 분석
    """
    
    def __init__(
        self,
        variable_importance: np.ndarray,
        feature_names: List[str],
        horizons: List[int]
    ):
        """
        Args:
            variable_importance: [num_samples, seq_length, num_features]
            feature_names: feature 이름 리스트
            horizons: 예측 horizon 리스트
        """
        self.variable_importance = variable_importance
        self.feature_names = feature_names
        self.horizons = horizons
        self.num_features = len(feature_names)
    
    def compute_average_importance(self) -> pd.DataFrame:
        """
        전체 샘플에 대한 평균 변수 중요도
        
        Returns:
            DataFrame with columns: feature, importance
        """
        # [num_samples, seq_length, num_features] -> [num_features]
        avg_importance = self.variable_importance.mean(axis=(0, 1))
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        })
        
        df = df.sort_values('importance', ascending=False)
        
        return df
    
    def compute_temporal_feature_importance(self) -> pd.DataFrame:
        """
        시점별 변수 중요도
        
        Returns:
            DataFrame with columns: timestep, feature, importance
        """
        seq_length = self.variable_importance.shape[1]
        
        # [num_samples, seq_length, num_features] -> [seq_length, num_features]
        temporal_importance = self.variable_importance.mean(axis=0)
        
        data = []
        for t in range(seq_length):
            for f, feature_name in enumerate(self.feature_names):
                data.append({
                    'timestep': t,
                    'feature': feature_name,
                    'importance': temporal_importance[t, f]
                })
        
        df = pd.DataFrame(data)
        
        return df
    
    def identify_top_features(self, top_k: int = 20) -> List[str]:
        """
        가장 중요한 상위 k개 feature 식별
        
        Args:
            top_k: 상위 몇 개
        
        Returns:
            상위 feature 이름 리스트
        """
        avg_importance = self.compute_average_importance()
        top_features = avg_importance.head(top_k)['feature'].tolist()
        
        return top_features
    
    def analyze_news_impact(self) -> Dict:
        """
        뉴스 관련 feature의 영향 분석
        
        Returns:
            dict with news-related analysis
        """
        # 뉴스 관련 feature 찾기
        news_features = [f for f in self.feature_names if 'news' in f.lower()]
        
        if len(news_features) == 0:
            return {
                'has_news_features': False,
                'message': 'No news features found'
            }
        
        # 뉴스 feature의 중요도
        avg_importance = self.compute_average_importance()
        news_importance = avg_importance[
            avg_importance['feature'].isin(news_features)
        ]
        
        # 전체 중요도 대비 뉴스 중요도 비율
        total_importance = avg_importance['importance'].sum()
        news_total_importance = news_importance['importance'].sum()
        news_ratio = news_total_importance / total_importance
        
        return {
            'has_news_features': True,
            'num_news_features': len(news_features),
            'news_features': news_features,
            'news_importance': news_importance,
            'news_importance_ratio': news_ratio,
            'top_news_feature': news_importance.iloc[0]['feature'] if len(news_importance) > 0 else None,
            'top_news_importance': news_importance.iloc[0]['importance'] if len(news_importance) > 0 else 0.0
        }
    
    def plot_top_features(
        self,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        상위 feature 중요도 시각화
        """
        avg_importance = self.compute_average_importance()
        top_features = avg_importance.head(top_k)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_temporal_heatmap(
        self,
        top_k: int = 20,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ):
        """
        시점별 feature 중요도 히트맵
        """
        # Top features 선택
        top_features = self.identify_top_features(top_k)
        
        # 해당 feature들의 temporal importance
        seq_length = self.variable_importance.shape[1]
        temporal_importance = self.variable_importance.mean(axis=0)  # [seq_length, num_features]
        
        # Top features만 추출
        feature_indices = [self.feature_names.index(f) for f in top_features]
        heatmap_data = temporal_importance[:, feature_indices]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_data.T,
            xticklabels=[f't-{seq_length-i-1}' for i in range(seq_length)],
            yticklabels=top_features,
            cmap='YlOrRd',
            cbar_kws={'label': 'Importance'}
        )
        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        plt.title('Temporal Feature Importance Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()


class AttentionAnalyzer:
    """
    TFT 모델의 Attention weight 분석
    """
    
    def __init__(
        self,
        attention_weights: np.ndarray,
        seq_length: int
    ):
        """
        Args:
            attention_weights: [num_samples, num_heads, seq_length, seq_length]
            seq_length: sequence length
        """
        self.attention_weights = attention_weights
        self.seq_length = seq_length
        self.num_heads = attention_weights.shape[1]
    
    def compute_average_attention(self) -> np.ndarray:
        """
        평균 attention weights
        
        Returns:
            avg_attention: [seq_length, seq_length]
        """
        # [num_samples, num_heads, seq_length, seq_length] -> [seq_length, seq_length]
        avg_attention = self.attention_weights.mean(axis=(0, 1))
        
        return avg_attention
    
    def compute_temporal_importance(self) -> np.ndarray:
        """
        각 시점의 중요도 (attention을 많이 받은 시점)
        
        Returns:
            temporal_importance: [seq_length]
        """
        # 각 시점이 받은 총 attention
        # [num_samples, num_heads, seq_length, seq_length] -> [seq_length]
        temporal_importance = self.attention_weights.mean(axis=(0, 1, 2))
        
        return temporal_importance
    
    def identify_key_timesteps(self, top_k: int = 5) -> List[int]:
        """
        가장 중요한 시점들 식별
        
        Args:
            top_k: 상위 몇 개
        
        Returns:
            중요한 시점 인덱스 리스트
        """
        temporal_importance = self.compute_temporal_importance()
        top_indices = np.argsort(temporal_importance)[-top_k:][::-1]
        
        return top_indices.tolist()
    
    def plot_attention_heatmap(
        self,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Attention weight 히트맵
        """
        avg_attention = self.compute_average_attention()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            avg_attention,
            xticklabels=[f't-{self.seq_length-i-1}' for i in range(self.seq_length)],
            yticklabels=[f't-{self.seq_length-i-1}' for i in range(self.seq_length)],
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Average Attention Weights')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_temporal_importance(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        시점별 중요도 그래프
        """
        temporal_importance = self.compute_temporal_importance()
        
        plt.figure(figsize=figsize)
        plt.bar(
            range(self.seq_length),
            temporal_importance,
            color='steelblue'
        )
        plt.xticks(
            range(self.seq_length),
            [f't-{self.seq_length-i-1}' for i in range(self.seq_length)],
            rotation=45
        )
        plt.xlabel('Time Step')
        plt.ylabel('Attention Importance')
        plt.title('Temporal Importance from Attention Weights')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal importance plot saved to {save_path}")
        
        plt.show()


class InterpretationReport:
    """
    종합 해석 리포트 생성
    """
    
    def __init__(
        self,
        feature_analyzer: FeatureImportanceAnalyzer,
        attention_analyzer: AttentionAnalyzer,
        horizons: List[int]
    ):
        self.feature_analyzer = feature_analyzer
        self.attention_analyzer = attention_analyzer
        self.horizons = horizons
    
    def generate_summary(self) -> Dict:
        """
        해석 결과 요약 생성
        
        Returns:
            summary dict
        """
        # Top features
        top_features = self.feature_analyzer.identify_top_features(top_k=10)
        
        # News impact
        news_analysis = self.feature_analyzer.analyze_news_impact()
        
        # Key timesteps
        key_timesteps = self.attention_analyzer.identify_key_timesteps(top_k=5)
        
        summary = {
            'top_10_features': top_features,
            'news_analysis': news_analysis,
            'key_timesteps': key_timesteps,
            'horizons': self.horizons
        }
        
        return summary
    
    def print_summary(self):
        """요약 출력"""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("INTERPRETATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Prediction Horizons: {summary['horizons']}")
        
        print("\n🔝 Top 10 Most Important Features:")
        for i, feature in enumerate(summary['top_10_features'], 1):
            print(f"  {i}. {feature}")
        
        news = summary['news_analysis']
        if news['has_news_features']:
            print(f"\n📰 News Feature Analysis:")
            print(f"  - Number of news features: {news['num_news_features']}")
            print(f"  - News importance ratio: {news['news_importance_ratio']:.2%}")
            print(f"  - Top news feature: {news['top_news_feature']}")
            print(f"  - Top news importance: {news['top_news_importance']:.4f}")
        else:
            print("\n📰 No news features found in the data")
        
        print(f"\n⏰ Key Timesteps (most attended):")
        for i, timestep in enumerate(summary['key_timesteps'], 1):
            print(f"  {i}. t-{self.attention_analyzer.seq_length - timestep - 1}")
        
        print("\n" + "="*60)
    
    def save_plots(self, output_dir: str):
        """모든 시각화 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance
        self.feature_analyzer.plot_top_features(
            top_k=20,
            save_path=str(output_dir / 'feature_importance.png')
        )
        
        # Temporal heatmap
        self.feature_analyzer.plot_temporal_heatmap(
            top_k=20,
            save_path=str(output_dir / 'temporal_feature_heatmap.png')
        )
        
        # Attention heatmap
        self.attention_analyzer.plot_attention_heatmap(
            save_path=str(output_dir / 'attention_heatmap.png')
        )
        
        # Temporal importance
        self.attention_analyzer.plot_temporal_importance(
            save_path=str(output_dir / 'temporal_importance.png')
        )
        
        print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    print("Interpretation module loaded successfully")
