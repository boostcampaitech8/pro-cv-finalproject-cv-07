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
        self.activation = nn.Sigmoid()

    def forward(self, image: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_img = self.backbone(image)
        if aux is None or self.fusion_mode == "none":
            z = z_img
        else:
            z = self.fusion(z_img, aux)
        out = self.head(z)
        return self.activation(out)
