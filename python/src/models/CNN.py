from typing import Optional

import torch
import torch.nn as nn

from src.models.fusion_modules import CrossAttentionFusion, GatedFusion, LateFusion, NoneFusion


_BACKBONE_MAP = {
    "convnext_tiny": "convnext_tiny",
    "resnet50": "resnet50",
    "vit_small": "vit_small_patch16_224",
}


class CNN(nn.Module):
    """
    CNN anomaly model using a timm backbone with optional aux fusion.

    FIXED ISSUES:
    1. Removed Sigmoid activation to prevent mode collapse
    2. Outputs are linear (no activation) for stable gradients
    3. Outputs are NOT clipped to [0, 1]

    This allows the model to:
    - Learn proper gradient flow
    - Predict varied severity scores
    - Avoid converging to constant predictions
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        in_chans: int = 3,
        aux_dim: int = 0,
        fusion: str = "none",
        dropout: float = 0.1,
        num_outputs: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone not in _BACKBONE_MAP:
            raise ValueError(f"Unsupported backbone: {backbone}")

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for CNN backbones. Install with: pip install timm"
            ) from exc

        model_name = _BACKBONE_MAP[backbone]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
        )

        self.embed_dim = getattr(self.backbone, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Backbone does not expose num_features")

        self.fusion_mode = fusion
        if fusion == "none":
            self.fusion = NoneFusion()
        elif fusion == "late":
            self.fusion = LateFusion(self.embed_dim, aux_dim, dropout=dropout)
        elif fusion == "gated":
            self.fusion = GatedFusion(self.embed_dim, aux_dim, dropout=dropout)
        elif fusion == "cross_attn":
            self.fusion = CrossAttentionFusion(self.embed_dim, aux_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion}")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_outputs),
        )

    def forward(self, image: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            torch.Tensor: Severity scores [B, 4] for horizons [1, 5, 10, 20]
                         Linear outputs (can be negative or > 1)
        """
        z_img = self.backbone(image)
        if aux is None or self.fusion_mode == "none":
            z = z_img
        else:
            z = self.fusion(z_img, aux)
        out = self.head(z)
        return out
