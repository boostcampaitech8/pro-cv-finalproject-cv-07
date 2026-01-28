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


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network
    각 feature의 중요도를 학습하여 선택
    """
    
    def __init__(
        self,
        input_dim: int,
        num_features: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.feature_embeddings = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) 
            for _ in range(num_features)
        ])
        
        # Feature-wise GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(num_features)
        ])
        
        # Variable selection weights
        self.softmax_grn = GatedResidualNetwork(
            input_dim=num_features * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_length, num_features, input_dim] 또는
               [batch_size, num_features, input_dim]
        
        Returns:
            selected_features: [batch_size, ..., hidden_dim]
            weights: [batch_size, ..., num_features] - Variable importance scores
        """
        # x shape 확인
        if len(x.shape) == 4:  # [batch_size, seq_length, num_features, input_dim]
            batch_size, seq_length, num_features, input_dim = x.shape
            
            # Feature-wise transformation
            transformed = []
            for i in range(num_features):
                feature_i = x[:, :, i, :]  # [batch_size, seq_length, input_dim]
                # Embedding으로 차원 축소
                embedded_i = self.feature_embeddings[i](feature_i)  # [batch_size, seq_length, hidden_dim]
                # GRN 적용
                transformed_i = self.feature_grns[i](embedded_i)
                transformed.append(transformed_i)
            
            # [batch_size, seq_length, num_features, hidden_dim]
            transformed = torch.stack(transformed, dim=2)
            
            # Flatten for weight computation
            # [batch_size, seq_length, num_features * hidden_dim]
            flattened = transformed.reshape(batch_size, seq_length, -1)
            
            # Compute selection weights
            # [batch_size, seq_length, num_features]
            weights = self.softmax(self.softmax_grn(flattened))
            
            # Weighted sum
            # [batch_size, seq_length, num_features, 1]
            weights_expanded = weights.unsqueeze(-1)
            # [batch_size, seq_length, hidden_dim]
            selected = (transformed * weights_expanded).sum(dim=2)
            
        else:  # [batch_size, num_features, input_dim]
            batch_size, num_features, input_dim = x.shape
            
            transformed = []
            for i in range(num_features):
                feature_i = x[:, i, :]
                transformed_i = self.feature_grns[i](feature_i)
                transformed.append(transformed_i)
            
            transformed = torch.stack(transformed, dim=1)
            flattened = transformed.reshape(batch_size, -1)
            weights = self.softmax(self.softmax_grn(flattened))
            weights_expanded = weights.unsqueeze(-1)
            selected = (transformed * weights_expanded).sum(dim=1)
        
        return selected, weights


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
    
    주요 특징:
    1. Variable Selection Network: 각 feature의 중요도 학습
    2. LSTM Encoder-Decoder: 시계열 패턴 학습
    3. Multi-head Attention: 시점별 중요도 학습
    4. Multi-horizon prediction: 여러 미래 시점 동시 예측
    """
    
    def __init__(
        self,
        num_features: int,
        num_horizons: int,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        use_variable_selection: bool = True,
        quantiles: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_horizons = num_horizons
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.use_variable_selection = use_variable_selection
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        
        # Variable Selection Network (optional)
        if use_variable_selection:
            self.variable_selection = VariableSelectionNetwork(
                input_dim=1,  # 각 feature는 scalar
                num_features=num_features,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            lstm_input_dim = hidden_dim
        else:
            lstm_input_dim = num_features
        
        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # LSTM Decoder (for future steps)
        self.lstm_decoder = nn.LSTM(
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
            return_attention: attention weights 반환 여부
        
        Returns:
            dict with keys:
                - 'predictions': [batch_size, num_horizons] or [batch_size, num_horizons, num_quantiles]
                - 'variable_importance': [batch_size, seq_length, num_features] (optional)
                - 'attention_weights': [batch_size, num_heads, seq_length, seq_length] (optional)
        """
        batch_size, seq_length, num_features = x.shape
        
        # Variable Selection
        if self.use_variable_selection:
            # Reshape for variable selection: [batch_size, seq_length, num_features, 1]
            x_reshaped = x.unsqueeze(-1)
            
            # Apply variable selection
            # selected: [batch_size, seq_length, hidden_dim]
            # var_weights: [batch_size, seq_length, num_features]
            selected_features, var_weights = self.variable_selection(x_reshaped)
            
            # Store for interpretation
            self.variable_importance = var_weights.detach()
            
            encoder_input = selected_features
        else:
            encoder_input = x
            var_weights = None
        
        # LSTM Encoding
        # encoder_output: [batch_size, seq_length, hidden_dim]
        encoder_output, (h_n, c_n) = self.lstm_encoder(encoder_input)
        
        # Multi-head Attention
        # Self-attention on encoder outputs
        attention_output, attention_weights = self.attention(
            query=encoder_output,
            key=encoder_output,
            value=encoder_output
        )
        
        # Store for interpretation
        self.temporal_importance = attention_weights.detach()
        
        # Post-attention GRN
        attention_output = self.post_attention_grn(attention_output)
        
        # Aggregate temporal information (use last timestep)
        # [batch_size, hidden_dim]
        aggregated = attention_output[:, -1, :]
        
        # Multi-horizon prediction
        predictions = []
        for i in range(self.num_horizons):
            pred_i = self.output_layers[i](aggregated)
            predictions.append(pred_i)
        
        # Stack predictions
        if self.quantiles:
            # [batch_size, num_horizons, num_quantiles]
            predictions = torch.stack(predictions, dim=1)
        else:
            # [batch_size, num_horizons]
            predictions = torch.cat(predictions, dim=1)
        
        # Prepare output
        output = {'predictions': predictions}
        
        if return_attention or self.variable_importance is not None:
            output['variable_importance'] = self.variable_importance
            output['attention_weights'] = self.temporal_importance
        
        return output
    
    def get_variable_importance(self) -> Optional[torch.Tensor]:
        """변수 중요도 반환"""
        return self.variable_importance
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Attention weights 반환"""
        return self.temporal_importance


# Quantile Loss for uncertainty estimation
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
        
        # Expand targets to match predictions
        # [batch_size, num_horizons, num_quantiles]
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)
        
        # Compute quantile loss
        errors = targets_expanded - predictions
        loss = 0.0
        
        for i, q in enumerate(self.quantiles):
            # Quantile loss for each quantile
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
    num_features = 50
    num_horizons = 4
    
    # 모델 생성
    model = TemporalFusionTransformer(
        num_features=num_features,
        num_horizons=num_horizons,
        hidden_dim=64,
        lstm_layers=2,
        attention_heads=4,
        dropout=0.1,
        use_variable_selection=True,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    # 더미 데이터
    x = torch.randn(batch_size, seq_length, num_features)
    
    # Forward pass
    output = model(x, return_attention=True)
    
    print("Model output shapes:")
    print(f"  Predictions: {output['predictions'].shape}")
    print(f"  Variable importance: {output['variable_importance'].shape}")
    print(f"  Attention weights: {output['attention_weights'].shape}")
    
    # Loss 계산
    targets = torch.randn(batch_size, num_horizons)
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    loss = criterion(output['predictions'], targets)
    print(f"\nQuantile Loss: {loss.item():.4f}")
