import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN)
    TFT의 핵심 building block
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # Layer 1
        if context_dim is not None:
            self.fc1 = nn.Linear(input_dim + context_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating layer
        self.gate = nn.Linear(hidden_dim, output_dim)
        
        # Skip connection
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, ..., input_dim]
            context: [batch_size, ..., context_dim] (optional)
        
        Returns:
            output: [batch_size, ..., output_dim]
        """
        # Concatenate context if provided
        if context is not None:
            x_concat = torch.cat([x, context], dim=-1)
        else:
            x_concat = x
        
        # Layer 1
        h = F.elu(self.fc1(x_concat))
        
        # Layer 2
        h = self.fc2(h)
        h = self.dropout(h)
        
        # Gating
        gate = torch.sigmoid(self.gate(h))
        h = gate * h
        
        # Skip connection
        if self.skip is not None:
            x = self.skip(x)
        
        # Residual + Layer Norm
        output = self.layer_norm(x + h)
        
        return output


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretability
    각 시점의 중요도를 파악할 수 있는 attention
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretation
        self.attention_weights = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len_q, embed_dim]
            key: [batch_size, seq_len_k, embed_dim]
            value: [batch_size, seq_len_v, embed_dim]
            mask: [batch_size, seq_len_q, seq_len_k] (optional)
        
        Returns:
            output: [batch_size, seq_len_q, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # Project and reshape
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store for interpretation
        self.attention_weights = attention_weights.detach()
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)
        
        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Multi-Horizon Stock Price Prediction
    뉴스 embedding을 별도로 처리하는 버전
    """
    
    def __init__(
        self,
        num_features: int,
        num_horizons: int,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        use_variable_selection: bool = False,
        quantiles: Optional[List[float]] = None,
        news_embedding_dim: int = 512,
        news_projection_dim: int = 32
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_horizons = num_horizons
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.use_variable_selection = use_variable_selection
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        
        self.news_embedding_dim = news_embedding_dim
        self.news_projection_dim = news_projection_dim
        
        # ===== 뉴스 Embedding Projection (Learnable!) =====
        print(f"Initializing news projection: {news_embedding_dim} → {news_projection_dim}")
        self.news_projection = nn.Sequential(
            nn.Linear(news_embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, news_projection_dim),
            nn.LayerNorm(news_projection_dim),
            nn.ReLU()
        )
        # =================================================
        
        # Feature 개수 체크
        if num_features < news_embedding_dim:
            # 뉴스 embedding이 포함되지 않은 경우
            print(f"⚠️  Warning: num_features ({num_features}) < news_embedding_dim ({news_embedding_dim})")
            print(f"   Assuming no news embeddings in data")
            self.has_news = False
            self.num_basic_features = num_features
            self.total_features = num_features
        else:
            # 뉴스 embedding이 포함된 경우
            self.has_news = True
            self.num_basic_features = num_features - news_embedding_dim
            self.total_features = self.num_basic_features + news_projection_dim
        
        print(f"Basic features: {self.num_basic_features}")
        print(f"News embedding: {news_embedding_dim if self.has_news else 0} → {news_projection_dim if self.has_news else 0}")
        print(f"Total features after projection: {self.total_features}")
        
        lstm_input_dim = self.total_features
        
        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head Attention
        self.attention = InterpretableMultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Output layers (per horizon)
        if quantiles:
            # Quantile regression
            self.output_layers = nn.ModuleList([
                nn.Linear(hidden_dim, len(quantiles)) for _ in range(num_horizons)
            ])
        else:
            # Point prediction
            self.output_layers = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_horizons)
            ])
        
        self.dropout = nn.Dropout(dropout)
        
        # For storing interpretation data
        self.variable_importance = None
        self.temporal_importance = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_length, num_features]
               마지막 512개가 news embedding (has_news=True인 경우)
        
        Returns:
            dict with predictions and attention weights
        """
        batch_size, seq_length, num_features = x.shape
        
        # ===== 1. 뉴스 embedding 분리 및 projection =====
        if self.has_news:
            # 기본 features (앞쪽)
            basic_features = x[:, :, :self.num_basic_features]  # [batch, seq, ~7]
            
            # 뉴스 embedding (뒤쪽 512개)
            news_embedding = x[:, :, self.num_basic_features:]  # [batch, seq, 512]
            
            # Learnable projection (512 → 32)
            news_projected = self.news_projection(news_embedding)  # [batch, seq, 32]
            
            # 합치기
            encoder_input = torch.cat([basic_features, news_projected], dim=-1)  # [batch, seq, ~39]
        else:
            # 뉴스 없으면 그대로
            encoder_input = x
        # ===============================================
        
        # ===== 2. LSTM Encoding =====
        encoder_output, (h_n, c_n) = self.lstm_encoder(encoder_input)
        # encoder_output: [batch_size, seq_length, hidden_dim]
        
        # ===== 3. Multi-head Attention =====
        attention_output, attention_weights = self.attention(
            query=encoder_output,
            key=encoder_output,
            value=encoder_output
        )
        
        # Store for interpretation
        self.temporal_importance = attention_weights.detach()
        
        # ===== 4. Post-attention GRN =====
        attention_output = self.post_attention_grn(attention_output)
        
        # ===== 5. Aggregate (마지막 timestep) =====
        aggregated = attention_output[:, -1, :]  # [batch_size, hidden_dim]
        
        # ===== 6. Multi-horizon prediction =====
        predictions = []
        for i in range(self.num_horizons):
            pred_i = self.output_layers[i](aggregated)
            predictions.append(pred_i)
        
        # Stack predictions
        if self.quantiles:
            predictions = torch.stack(predictions, dim=1)  # [batch, horizons, quantiles]
        else:
            predictions = torch.cat(predictions, dim=1)  # [batch, horizons]
        
        # ===== Output =====
        output = {'predictions': predictions}
        
        if return_attention:
            output['attention_weights'] = self.temporal_importance
            output['variable_importance'] = None  # Variable selection 꺼져있음
        
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Attention weights 반환"""
        return self.temporal_importance


# Quantile Loss
class QuantileLoss(nn.Module):
    """
    Quantile Loss for probabilistic forecasting
    """
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_horizons, num_quantiles]
            targets: [batch_size, num_horizons]
        
        Returns:
            loss: scalar
        """
        batch_size, num_horizons, num_quantiles = predictions.shape
        
        # Expand targets
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)
        
        # Compute quantile loss
        errors = targets_expanded - predictions
        loss = 0.0
        
        for i, q in enumerate(self.quantiles):
            q_loss = torch.max(
                q * errors[:, :, i],
                (q - 1) * errors[:, :, i]
            )
            loss += q_loss.mean()
        
        return loss / len(self.quantiles)


if __name__ == "__main__":
    # 테스트
    batch_size = 32
    seq_length = 20
    num_basic_features = 7
    news_embedding_dim = 512
    num_features = num_basic_features + news_embedding_dim  # 519
    num_horizons = 4
    
    print(f"Testing TFT with news embedding...")
    print(f"Input features: {num_features} (basic: {num_basic_features}, news: {news_embedding_dim})")
    
    # 모델 생성
    model = TemporalFusionTransformer(
        num_features=num_features,
        num_horizons=num_horizons,
        hidden_dim=64,
        lstm_layers=2,
        attention_heads=4,
        dropout=0.1,
        use_variable_selection=False,
        quantiles=[0.1, 0.5, 0.9],
        news_embedding_dim=512,
        news_projection_dim=32
    )
    
    # 더미 데이터
    x = torch.randn(batch_size, seq_length, num_features)
    
    # Forward pass
    output = model(x, return_attention=True)
    
    print("\nModel output shapes:")
    print(f"  Predictions: {output['predictions'].shape}")
    print(f"  Attention weights: {output['attention_weights'].shape}")
    
    # Loss 계산
    targets = torch.randn(batch_size, num_horizons)
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    loss = criterion(output['predictions'], targets)
    print(f"\nQuantile Loss: {loss.item():.4f}")
    
    print("\n✓ TFT with news projection working!")
