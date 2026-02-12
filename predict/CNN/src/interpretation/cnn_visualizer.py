from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch


HORIZONS = [1, 5, 10, 20]


def _to_numpy(array_like) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _load_image_as_array(image: Union[str, Path, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load image files.") from exc
        with Image.open(image) as img:
            return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    array = _to_numpy(image).astype(np.float32)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def _load_grayscale(image: Union[str, Path, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to load image files.") from exc
        with Image.open(image) as img:
            return np.asarray(img.convert("L"), dtype=np.float32)
    array = _to_numpy(image).astype(np.float32)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3:
        array = array.mean(axis=-1)
    return array


def _ensure_parent(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _extract_threshold(q_values, idx: int, horizon: int) -> Optional[float]:
    if q_values is None:
        return None
    if isinstance(q_values, dict):
        if horizon in q_values:
            return float(q_values[horizon])
        key = f"h{horizon}"
        if key in q_values:
            return float(q_values[key])
    values = _to_numpy(q_values).astype(float)
    if values.ndim == 0:
        return float(values)
    if idx < values.shape[0]:
        return float(values[idx])
    return None


def plot_severity_timeline(
    severity: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    actual: Optional[Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]]] = None,
    dates: Optional[Sequence[str]] = None,
    q80: Optional[Union[Sequence[float], Dict]] = None,
    q90: Optional[Union[Sequence[float], Dict]] = None,
    q95: Optional[Union[Sequence[float], Dict]] = None,
    output_path: Union[str, Path] = "severity_timeline.png",
    title: str = "Severity Timeline",
    figsize: Tuple[int, int] = (12, 8),
) -> str:
    """
    Plot per-horizon severity curves with q80/q90/q95 reference lines.
    """
    import matplotlib.pyplot as plt

    severity_np = _to_numpy(severity).astype(np.float32)
    if severity_np.ndim == 1:
        severity_np = severity_np[None, :]
    if severity_np.shape[1] != len(HORIZONS):
        raise ValueError("Severity must have shape [N, 4] for horizons [1,5,10,20].")

    actual_np: Optional[np.ndarray] = None
    if actual is not None:
        actual_np = _to_numpy(actual).astype(np.float32)
        if actual_np.ndim == 1:
            actual_np = actual_np[None, :]
        if actual_np.shape[1] != len(HORIZONS):
            raise ValueError("Actual returns must have shape [N, 4] for horizons [1,5,10,20].")

    num_points = severity_np.shape[0]
    x_values = list(range(num_points)) if dates is None else list(dates)

    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=figsize, sharex=True)
    if len(HORIZONS) == 1:
        axes = [axes]

    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx]
        ax.plot(x_values, severity_np[:, idx], label=f"pred_h{horizon}", color="tab:blue")
        if actual_np is not None:
            ax.plot(
                x_values,
                actual_np[:, idx],
                label=f"actual_h{horizon}",
                color="tab:green",
                alpha=0.6,
            )

        q80_val = _extract_threshold(q80, idx, horizon)
        q90_val = _extract_threshold(q90, idx, horizon)
        q95_val = _extract_threshold(q95, idx, horizon)

        if q80_val is not None:
            ax.axhline(q80_val, color="gray", linestyle="--", linewidth=1, label="q80")
        if q90_val is not None:
            ax.axhline(q90_val, color="orange", linestyle="--", linewidth=1, label="q90")
        if q95_val is not None:
            ax.axhline(q95_val, color="red", linestyle="--", linewidth=1, label="q95")

        ax.set_ylabel(f"h{horizon}")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    if dates is not None:
        # Reduce tick density for readability
        max_ticks = 12
        if num_points > max_ticks:
            step = max(1, num_points // max_ticks)
            tick_idx = list(range(0, num_points, step))
            if tick_idx[-1] != num_points - 1:
                tick_idx.append(num_points - 1)
            tick_vals = [x_values[i] for i in tick_idx]
            for ax in axes:
                ax.set_xticks(tick_vals)
        for ax in axes:
            for label in ax.get_xticklabels():
                label.set_rotation(90)
                label.set_ha("right")
                label.set_fontsize(8)

    axes[-1].set_xlabel("Date" if dates is not None else "Index")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def overlay_gradcam_on_candlestick(
    candlestick_image: Union[str, Path, np.ndarray, torch.Tensor],
    gradcam: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path] = "gradcam_overlay.png",
    alpha: float = 0.4,
    cmap: str = "jet",
) -> str:
    """
    Overlay Grad-CAM heatmap on the original candlestick image.
    """
    import matplotlib.cm as cm
    from PIL import Image

    base = _load_image_as_array(candlestick_image)
    heatmap = _to_numpy(gradcam).astype(np.float32)

    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()
    if heatmap.ndim != 2:
        raise ValueError("Grad-CAM input must be a 2D heatmap.")

    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((base.shape[1], base.shape[0]), resample=Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    colormap = cm.get_cmap(cmap)
    colored = colormap(heatmap_resized)[..., :3]

    blended = (1 - alpha) * base + alpha * colored
    blended = np.clip(blended, 0.0, 1.0)

    output_path = _ensure_parent(output_path)
    Image.fromarray((blended * 255).astype(np.uint8)).save(output_path)
    return str(output_path)


def plot_gaf_rp_heatmap(
    image: Union[str, Path, np.ndarray, torch.Tensor],
    output_path: Union[str, Path] = "gaf_rp_heatmap.png",
    title: str = "GAF/RP Heatmap",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (6, 6),
) -> str:
    """
    Plot GAF or RP image as a heatmap.
    """
    import matplotlib.pyplot as plt

    array = _load_grayscale(image)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(array, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")

    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_fusion_contribution(
    fusion: str,
    output_path: Union[str, Path] = "fusion_contribution.png",
    gate: Optional[Union[np.ndarray, torch.Tensor, float]] = None,
    image_score: Optional[float] = None,
    aux_score: Optional[float] = None,
    attention_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 4),
) -> str:
    """
    Visualize fusion contributions for gated or cross-attention fusion.
    """
    import matplotlib.pyplot as plt

    fusion = fusion.lower()
    if fusion == "gated":
        if image_score is None or aux_score is None:
            if gate is None:
                raise ValueError("Provide gate or image_score/aux_score for gated fusion plot.")
            gate_values = _to_numpy(gate).astype(np.float32)
            gate_mean = float(np.mean(gate_values))
            image_score = gate_mean
            aux_score = 1.0 - gate_mean

        labels = ["image", "aux"]
        values = [float(image_score), float(aux_score)]
        plot_title = title or "Gated Fusion Contribution"

    elif fusion == "cross_attn":
        if attention_weights is None:
            raise ValueError("attention_weights required for cross-attn plot.")
        weights = _to_numpy(attention_weights).astype(np.float32)
        if weights.ndim == 0:
            weights = weights[None]
        if weights.ndim > 1:
            weights = weights.mean(axis=tuple(range(weights.ndim - 1)))
        weights = np.maximum(weights, 0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        labels = [f"aux_{i}" for i in range(weights.shape[0])]
        values = weights.tolist()
        plot_title = title or "Cross-Attention Summary"

    else:
        raise ValueError("fusion must be 'gated' or 'cross_attn'.")
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color="tab:blue")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Contribution")
    ax.set_title(plot_title)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    output_path = _ensure_parent(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


class GradCAM:
    """
    Lightweight Grad-CAM helper for CNN interpretability.

    FIXED ISSUES:
    1. Hooks re-registered for each generate() call
    2. Activations/gradients reset each time
    3. Hook cleanup to prevent memory leaks
    4. Per-sample normalization
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        method: str = "gradcam",
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.method = method.lower().strip()
        if self.method not in {"gradcam", "layercam"}:
            raise ValueError("GradCAM method must be 'gradcam' or 'layercam'.")
        self._activations = None
        self._gradients = None
        self._handles = []

    def _clear_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register_hooks(self) -> None:
        self._clear_hooks()

        def forward_hook(_, __, output):
            self._activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.append(handle_fwd)
        self._handles.append(handle_bwd)

    def generate(
        self,
        input_tensor: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a Grad-CAM heatmap for a given input tensor.
        """
        self._activations = None
        self._gradients = None
        self._register_hooks()

        self.model.zero_grad()
        input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor, aux)

        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1)[0])

        score = outputs[:, class_idx].sum()
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients."
            )

        if self.method == "gradcam":
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self._activations).sum(dim=1)
        else:
            # LayerCAM: emphasize spatially localized positive gradients
            cam = torch.relu(self._gradients) * self._activations
            cam = cam.sum(dim=1)
        cam = torch.relu(cam)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        self._clear_hooks()
        return cam

    def __del__(self) -> None:
        self._clear_hooks()
