import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_cnn import CNNDataset, VOLUME_COLUMNS, cnn_collate_fn
from src.interpretation.cnn_visualizer import GradCAM
from src.models.CNN import CNN


HORIZONS = [1, 5, 10, 20]


def _find_last_conv(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    for _, layer in reversed(list(module.named_modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    return None


def _build_severity_levels(
    severity_scores: np.ndarray,
    q80: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
) -> Dict[str, str]:
    """
    Map severity scores to string levels using dataset q80/q90/q95 thresholds.
    """
    levels: Dict[str, str] = {}
    denom = np.maximum(q95 - q80, 1e-8)
    s90 = np.clip((q90 - q80) / denom, 0.0, 1.0)

    for idx, horizon in enumerate(HORIZONS):
        score = float(severity_scores[idx])
        if score <= 0.0:
            level = "below_q80"
        elif score < float(s90[idx]):
            level = "q80_q90"
        elif score < 1.0:
            level = "q90_q95"
        else:
            level = "above_q95"
        levels[f"h{horizon}"] = level

    return levels


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def run_inference_cnn(
    commodity: str,
    fold: int,
    window_size: int,
    image_mode: str,
    backbone: str,
    fusion: str,
    exp_name: str,
    use_aux: bool = False,
    aux_type: str = "volume",
    checkpoint_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None,
    save_gradcam: bool = False,
) -> Path:
    """
    Run inference on the validation split and save per-date JSON outputs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CNNDataset(
        commodity=commodity,
        fold=fold,
        split="val",
        window_size=window_size,
        image_mode=image_mode,
        use_aux=use_aux,
        aux_type=aux_type,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cnn_collate_fn,
    )

    in_chans = 3 if image_mode == "stack" else 1

    aux_dim = 0
    if use_aux:
        if aux_type in {"volume", "both"}:
            aux_dim += len(VOLUME_COLUMNS)
        if aux_type in {"news", "both"}:
            aux_dim += dataset.news_dim

    model = CNN(
        backbone=backbone,
        in_chans=in_chans,
        aux_dim=aux_dim,
        fusion=fusion,
        dropout=0.1,
        num_outputs=4,
        pretrained=False,
    )

    if checkpoint_path is None:
        checkpoint_path = (
            Path("src/outputs/checkpoints/cnn")
            / commodity
            / exp_name
            / f"fold_{fold}"
            / "best.pt"
        )
    checkpoint_path = Path(checkpoint_path)
    _load_checkpoint(model, checkpoint_path, device)

    model.to(device)
    model.eval()

    results_root = (
        Path("src/outputs/predictions/cnn")
        / commodity
        / exp_name
        / f"fold_{fold}"
        / "results"
    )
    results_root.mkdir(parents=True, exist_ok=True)

    q80 = np.asarray(dataset.q80, dtype=np.float32)
    q90 = np.asarray(dataset.q90, dtype=np.float32)
    q95 = np.asarray(dataset.q95, dtype=np.float32)

    gradcam = None
    gradcam_root = results_root / "gradcam"
    if save_gradcam:
        target_layer = _find_last_conv(model.backbone)
        if target_layer is not None:
            gradcam = GradCAM(model, target_layer)
            gradcam_root.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        images = batch["image"].to(device)
        aux = batch.get("aux")
        if aux is not None:
            aux = aux.to(device)

        with torch.no_grad():
            outputs = model(images, aux).detach().cpu().numpy()

        meta = batch["meta"]
        dates = meta["date"]

        for i, date_str in enumerate(dates):
            severity_scores = outputs[i]
            severity = {
                "h1": float(severity_scores[0]),
                "h5": float(severity_scores[1]),
                "h10": float(severity_scores[2]),
                "h20": float(severity_scores[3]),
            }
            severity_level = _build_severity_levels(severity_scores, q80, q90, q95)

            payload = {
                "meta": {
                    "date": date_str,
                    "commodity": commodity,
                    "window": window_size,
                    "image_mode": image_mode,
                    "aux_type": aux_type if use_aux else "none",
                    "fusion": fusion,
                    "backbone": backbone,
                    "fold": fold,
                },
                "scores": {
                    "severity": severity,
                    "severity_level": severity_level,
                },
            }

            output_path = results_root / f"{date_str}.json"
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        if gradcam is not None:
            for i, date_str in enumerate(dates):
                img_i = images[i : i + 1]
                aux_i = aux[i : i + 1] if aux is not None else None
                for h_idx, horizon in enumerate(HORIZONS):
                    with torch.enable_grad():
                        cam = gradcam.generate(img_i, aux=aux_i, class_idx=h_idx)
                    cam_np = cam.detach().cpu().numpy()[0]
                    out_dir = gradcam_root / f"h{horizon}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / f"{date_str}.npy", cam_np)

    return results_root
