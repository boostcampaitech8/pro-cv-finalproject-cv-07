from typing import Optional

import torch
import torch.nn as nn


class NoneFusion(nn.Module):
    """
    Return image embedding without fusion.
    """

    def forward(self, z_img: torch.Tensor, z_aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        return z_img


class LateFusion(nn.Module):
    """
    Concatenate image and aux embeddings, then project to image embedding space.
    """

    def __init__(
        self,
        img_dim: int,
        aux_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or img_dim
        self.aux_proj = nn.Linear(aux_dim, img_dim) if aux_dim > 0 else None
        self.fusion = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, img_dim),
        )

    def forward(self, z_img: torch.Tensor, z_aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z_aux is None or self.aux_proj is None:
            return z_img
        z_aux = self.aux_proj(z_aux)
        fused = torch.cat([z_img, z_aux], dim=-1)
        return self.fusion(fused)


class GatedFusion(nn.Module):
    """
    Gated fusion between image and aux embeddings.
    """

    def __init__(
        self,
        img_dim: int,
        aux_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or img_dim
        self.aux_proj = nn.Linear(aux_dim, img_dim) if aux_dim > 0 else None
        self.gate = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, img_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_img: torch.Tensor, z_aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z_aux is None or self.aux_proj is None:
            return z_img
        z_aux = self.aux_proj(z_aux)
        gate = self.gate(torch.cat([z_img, z_aux], dim=-1))
        return gate * z_img + (1.0 - gate) * z_aux


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion where image embedding queries aux embedding.
    """

    def __init__(
        self,
        img_dim: int,
        aux_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aux_proj = nn.Linear(aux_dim, img_dim) if aux_dim > 0 else None
        self.attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, z_img: torch.Tensor, z_aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z_aux is None or self.aux_proj is None:
            return z_img

        query = z_img.unsqueeze(1)  # [B, 1, D]
        key_value = self.aux_proj(z_aux).unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attn(query, key_value, key_value)
        return attn_out.squeeze(1) + z_img
