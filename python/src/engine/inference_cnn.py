import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pandas.tseries.offsets import BDay

from src.data.dataset_cnn import CNNDataset, cnn_collate_fn
from src.interpretation.cnn_visualizer import GradCAM
from src.models.CNN import CNN
from src.utils.unified_output import build_anomaly_payload, map_severity_levels, write_json


HORIZONS = list(range(1, 21))


def _find_last_conv(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    for _, layer in reversed(list(module.named_modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    return None


def _select_gradcam_layer(
    backbone: torch.nn.Module,
    backbone_name: str,
    stage_index: Optional[int] = None,
) -> Optional[torch.nn.Module]:
    """
    Select a stable, spatially-aware conv layer for Grad-CAM.
    """
    name = backbone_name.lower()

    if "convnext" in name:
        stages = getattr(backbone, "stages", None)
        if stages:
            # Prefer higher-resolution stage if provided, else use penultimate stage.
            if stage_index is not None:
                stage_indices = [stage_index]
            else:
                stage_indices = [-2, -1] if len(stages) >= 2 else [-1]
            for s_idx in stage_indices:
                stage = stages[s_idx]
                try:
                    last_block = stage[-1]
                except Exception:
                    last_block = stage
                for attr in ("dwconv", "conv_dw", "conv"):
                    if hasattr(last_block, attr):
                        layer = getattr(last_block, attr)
                        if isinstance(layer, torch.nn.Conv2d):
                            return layer

    if "resnet" in name:
        layer4 = getattr(backbone, "layer4", None)
        if layer4:
            block = layer4[-1]
            for attr in ("conv3", "conv2", "conv1"):
                if hasattr(block, attr):
                    layer = getattr(block, attr)
                    if isinstance(layer, torch.nn.Conv2d):
                        return layer

    # ViT-style backbones do not have conv feature maps for standard Grad-CAM.
    if "vit" in name:
        return None

    return _find_last_conv(backbone)


def _cam_stats(cam: np.ndarray) -> Dict[str, float]:
    cam_min = float(np.min(cam))
    cam_max = float(np.max(cam))
    cam_mean = float(np.mean(cam))
    cam_std = float(np.std(cam))
    return {
        "min": cam_min,
        "max": cam_max,
        "mean": cam_mean,
        "std": cam_std,
    }


def _auto_select_gradcam_stage(
    model: torch.nn.Module,
    backbone_name: str,
    images: torch.Tensor,
    aux: Optional[torch.Tensor],
    candidate_stages: List[int],
    gradcam_method: str,
) -> Optional[int]:
    """
    Pick a Grad-CAM stage that maximizes average CAM std on a small sample.
    This tends to choose a layer with higher spatial variation.
    """
    if images.numel() == 0:
        return None

    sample_count = min(4, images.shape[0])
    best_stage = None
    best_score = -1.0

    for stage_idx in candidate_stages:
        layer = _select_gradcam_layer(model.backbone, backbone_name, stage_idx)
        if layer is None:
            continue
        cam_helper = GradCAM(model, layer, method=gradcam_method)
        cams = []
        for i in range(sample_count):
            img_i = images[i : i + 1]
            aux_i = aux[i : i + 1] if aux is not None else None
            with torch.enable_grad():
                cam = cam_helper.generate(img_i, aux=aux_i, class_idx=0)
            cams.append(cam.detach().cpu().numpy()[0])
        if not cams:
            continue
        std_score = float(np.mean([np.std(c) for c in cams]))
        ref = cams[0]
        diff_score = float(np.mean([np.mean(np.abs(c - ref)) for c in cams[1:]]) if len(cams) > 1 else 0.0)
        score = std_score + diff_score
        if score > best_score:
            best_score = score
            best_stage = stage_idx

    return best_stage


def _build_severity_levels_from_values(
    values: np.ndarray,
    q80: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
) -> Dict[str, str]:
    """
    Map values to severity levels using dataset thresholds.
    """
    return map_severity_levels(values, q80, q90, q95, HORIZONS)


def _build_severity_levels(
    raw_returns: np.ndarray,
    q80: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
) -> Dict[str, str]:
    """
    Map raw return values to severity levels using dataset thresholds.
    """
    return _build_severity_levels_from_values(raw_returns, q80, q90, q95)


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.name == "best_model.pt":
        alt = path.with_name("best.pt")
        if alt.exists():
            return alt
    if path.name == "best.pt":
        alt = path.with_name("best_model.pt")
        if alt.exists():
            return alt
    return path


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def _infer_date_tag(split_path: Optional[Path]) -> str:
    if split_path is None or not split_path.exists():
        return "latest"
    try:
        payload = json.loads(split_path.read_text())
        meta = payload.get("meta", {})
        if isinstance(meta, dict):
            inference_window = meta.get("inference_window", {})
            for key in ("end", "end_date"):
                if key in inference_window:
                    return str(inference_window[key])[:10]
            for key in ("test_end_date", "end_date"):
                if key in meta:
                    return str(meta[key])[:10]
    except Exception:
        pass
    return "latest"


def _next_trading_dates(
    as_of: str,
    horizons: Sequence[int],
    trading_dates: Sequence[str],
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if as_of in trading_dates:
        idx = trading_dates.index(as_of)
        future = trading_dates[idx + 1 :]
        for h in horizons:
            if h - 1 < len(future):
                mapping[h] = future[h - 1]
    base = pd.to_datetime(as_of)
    for h in horizons:
        if h not in mapping:
            mapping[h] = (base + BDay(h)).strftime("%Y-%m-%d")
    return mapping


def run_inference_cnn(
    commodity: str,
    fold: int,
    window_size: int,
    image_mode: str,
    backbone: str,
    fusion: str,
    exp_name: str,
    split: str = "val",
    use_aux: bool = False,
    aux_type: str = "news",
    checkpoint_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None,
    save_gradcam: bool = False,
    gradcam_stage: Optional[int] = None,
    gradcam_method: str = "gradcam",
    save_legacy: bool = False,
    save_unified: bool = False,
    data_dir: Optional[str] = None,
    split_file: Optional[str] = None,
    prediction_root: Optional[str] = None,
    date_tag: Optional[str] = None,
    latest_only: bool = True,
    write_json: bool = False,
    write_csv: bool = True,
) -> Path:
    """
    Run inference on the requested split and save per-date JSON outputs.
    Legacy outputs go to src/outputs/predictions/cnn, unified outputs to
    src/outputs/predictions/unified/cnn.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    split_path = Path(split_file) if split_file else None
    if split_path is not None and not split_path.is_absolute() and data_dir:
        split_path = Path(data_dir) / split_path

    dataset = CNNDataset(
        commodity=commodity,
        fold=fold,
        split=split,
        window_size=window_size,
        image_mode=image_mode,
        use_aux=use_aux,
        aux_type=aux_type,
        data_dir=data_dir,
        split_file=str(split_path) if split_path else None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cnn_collate_fn,
    )

    in_chans = 3 if image_mode == "stack" else 2 if image_mode in {"candle_gaf", "candle_rp"} else 1
    aux_dim = 0
    if use_aux:
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

    date_tag = date_tag or _infer_date_tag(split_path)
    if prediction_root:
        pred_root = Path(
            prediction_root.format(
                commodity=commodity,
                date=date_tag,
            )
        )
    else:
        pred_root = (
            Path("src/outputs/predictions/cnn")
            / commodity
            / exp_name
            / f"fold_{fold}"
        )

    results_root = pred_root / f"w{window_size}" / "results"
    if write_json and save_legacy:
        results_root.mkdir(parents=True, exist_ok=True)

    unified_root = pred_root / f"w{window_size}" / "results_unified"
    if write_json and save_unified:
        unified_root.mkdir(parents=True, exist_ok=True)

    q80 = np.asarray(dataset.q80, dtype=np.float32)
    q90 = np.asarray(dataset.q90, dtype=np.float32)
    q95 = np.asarray(dataset.q95, dtype=np.float32)

    is_infer = split == "infer"
    raw_returns_dict: Dict[str, np.ndarray] = {}
    if not is_infer:
        for idx, anchor_idx in enumerate(dataset.anchor_indices):
            date = dataset.anchor_dates[idx]
            raw_returns_dict[date] = dataset.anchor_returns[idx]

    gradcam = None
    gradcam_root = results_root / "gradcam"
    gradcam_stats_path = None
    gradcam_stats_fh = None

    # Prepare iterator so we can inspect the first batch for auto stage selection.
    batch_iter = iter(dataloader)
    first_batch = next(batch_iter, None)
    if first_batch is None:
        return results_root

    if save_gradcam and write_json:
        if gradcam_stage is None and "convnext" in backbone.lower():
            images = first_batch["image"].to(device)
            aux = first_batch.get("aux")
            if aux is not None:
                aux = aux.to(device)
            auto_stage = _auto_select_gradcam_stage(
                model,
                backbone,
                images,
                aux,
                candidate_stages=[-4, -3, -2, -1],
                gradcam_method=gradcam_method,
            )
            gradcam_stage = auto_stage

        target_layer = _select_gradcam_layer(model.backbone, backbone, gradcam_stage)
        if target_layer is not None:
            gradcam = GradCAM(model, target_layer, method=gradcam_method)
            gradcam_root.mkdir(parents=True, exist_ok=True)
            gradcam_stats_path = gradcam_root / "gradcam_stats.jsonl"
            gradcam_stats_fh = gradcam_stats_path.open("w", encoding="utf-8")
            print(
                "Grad-CAM target layer:",
                target_layer.__class__.__name__,
                "stage:",
                gradcam_stage,
                "method:",
                gradcam_method,
            )
        else:
            print("Warning: Grad-CAM skipped (no suitable conv layer found).")

    csv_rows: List[Dict[str, str]] = []
    latest_date = max(dataset.anchor_dates) if dataset.anchor_dates else None

    def _process_batch(batch, batch_idx: int) -> int:
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
            raw_returns = None
            if not is_infer:
                raw_returns = raw_returns_dict.get(date_str)
                if raw_returns is None:
                    global_idx = batch_idx * batch_size + i
                    if global_idx < len(dataset.anchor_returns):
                        raw_returns = dataset.anchor_returns[global_idx]
                    else:
                        raw_returns = np.zeros(4, dtype=np.float32)

            severity = {
                f"h{h}": float(severity_scores[i])
                for i, h in enumerate(HORIZONS)
            }
            severity_level_pred = _build_severity_levels_from_values(severity_scores, q80, q90, q95)
            severity_level = None
            if raw_returns is not None:
                severity_level = _build_severity_levels(raw_returns, q80, q90, q95)

            if write_csv and (not latest_only or date_str == latest_date):
                date_map = _next_trading_dates(date_str, HORIZONS, dataset.dates)
                for h in HORIZONS:
                    csv_rows.append(
                        {
                            "date": date_str,
                            "window": str(window_size),
                            "predict_date": date_map.get(h),
                            "severity_level": severity_level_pred.get(f"h{h}", "normal"),
                        }
                    )

            if write_json and save_legacy:
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
                        "severity_level_pred": severity_level_pred,
                    },
                }
                if severity_level is not None and raw_returns is not None:
                    payload["scores"]["severity_level"] = severity_level
                    payload["scores"]["raw_returns"] = {
                        f"h{h}": float(raw_returns[i])
                        for i, h in enumerate(HORIZONS)
                    }

                output_path = results_root / f"{date_str}.json"
                output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

            if write_json and save_unified:
                unified_payload = build_anomaly_payload(
                    model="cnn",
                    commodity=commodity,
                    window=window_size,
                    horizons=HORIZONS,
                    as_of=date_str,
                    fold=fold,
                    scores=severity_scores,
                    q80=q80,
                    q90=q90,
                    q95=q95,
                    raw_returns=raw_returns,
                    model_variant=exp_name,
                    extra_meta={
                        "image_mode": image_mode,
                        "aux_type": aux_type if use_aux else "none",
                        "fusion": fusion,
                        "backbone": backbone,
                    },
                )
                unified_path = unified_root / f"{date_str}.json"
                write_json(unified_payload, unified_path)

        if gradcam is not None and write_json:
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
                    if gradcam_stats_fh is not None:
                        stats = _cam_stats(cam_np)
                        pred_val = float(outputs[i][h_idx])
                        gradcam_stats_fh.write(
                            json.dumps(
                                {
                                    "date": date_str,
                                    "horizon": horizon,
                                    "pred": pred_val,
                                    "stage": gradcam_stage,
                                    "method": gradcam_method,
                                    "cam": stats,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        return batch_idx + 1

    batch_idx = 0
    batch_idx = _process_batch(first_batch, batch_idx)
    for batch in batch_iter:
        batch_idx = _process_batch(batch, batch_idx)

    if gradcam_stats_fh is not None:
        gradcam_stats_fh.close()

    if write_csv:
        pred_root.mkdir(parents=True, exist_ok=True)
        csv_path = pred_root / "cnn_predictions.csv"
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df = df.dropna(subset=["predict_date"]).reset_index(drop=True)
            df.to_csv(csv_path, index=False)
    return pred_root
