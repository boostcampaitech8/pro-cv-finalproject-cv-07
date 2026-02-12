import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from metrics.metrics_cnn import HORIZONS, compute_metrics_per_horizon, summarize_metrics


def _compute_weighted_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
    weights: Optional[List[float]] = None,
    severity_weight: float = 0.0,
) -> torch.Tensor:
    loss = loss_fn(outputs, targets)  # [B, 4]
    if weights is not None:
        weight_tensor = torch.tensor(weights, device=loss.device, dtype=loss.dtype)
        loss = loss * weight_tensor
    if severity_weight > 0:
        sample_weight = 1.0 + severity_weight * targets.mean(dim=1, keepdim=True)
        loss = loss * sample_weight
    return loss.mean()


def _collect_raw_returns(dataset) -> Optional[np.ndarray]:
    if hasattr(dataset, "anchor_returns"):
        return np.asarray(dataset.anchor_returns, dtype=np.float32)
    return None


def _sanitize_for_json(obj):
    if isinstance(obj, (float, np.floating)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _build_binary_labels(
    dataset,
    threshold_key: str,
) -> Optional[np.ndarray]:
    raw_returns = _collect_raw_returns(dataset)
    if raw_returns is None:
        return None
    thresholds = getattr(dataset, threshold_key, None)
    if thresholds is None:
        return None
    thresholds = np.asarray(thresholds, dtype=np.float32)
    return (raw_returns >= thresholds).astype(int)


def _compute_pos_stats(y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for idx, horizon in enumerate(HORIZONS):
        labels = y_true[:, idx]
        pos_count = int(labels.sum())
        total = int(labels.shape[0])
        pos_rate = float(pos_count / total) if total > 0 else 0.0
        stats[str(horizon)] = {
            "pos_count": pos_count,
            "total": total,
            "pos_rate": pos_rate,
        }
    return stats


def evaluate_cnn(
    model: torch.nn.Module,
    dataloader,
    loss_fn,
    device: str,
    horizon_weights: Optional[List[float]] = None,
    severity_loss_weight: float = 0.0,
    threshold: float = 0.5,
    best_threshold: bool = False,
) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    losses: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            aux = batch.get("aux")
            if aux is not None:
                aux = aux.to(device)
            targets = batch["severity"].to(device)

            outputs = model(images, aux)
            loss = _compute_weighted_loss(
                outputs,
                targets,
                loss_fn,
                horizon_weights,
                severity_weight=severity_loss_weight,
            )
            losses.append(loss.item())

            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    if not all_preds:
        return float("nan"), {}

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # q95 labels (primary)
    y_true_q95 = _build_binary_labels(dataloader.dataset, "q95")
    if y_true_q95 is None or y_true_q95.shape[0] != preds.shape[0]:
        y_true_q95 = (targets >= 1.0).astype(int)

    thresholds_q95 = None
    if hasattr(dataloader.dataset, "q95"):
        thresholds_q95 = np.asarray(dataloader.dataset.q95, dtype=np.float32)
        if thresholds_q95.shape[0] != preds.shape[1]:
            thresholds_q95 = None

    metrics = compute_metrics_per_horizon(
        y_true_q95,
        preds,
        threshold=threshold,
        best_threshold=best_threshold,
        thresholds=thresholds_q95,
    )
    metrics_best = compute_metrics_per_horizon(
        y_true_q95,
        preds,
        threshold=threshold,
        best_threshold=True,
        thresholds=None,
    )
    for horizon in metrics:
        metrics[horizon]["f1_best"] = metrics_best[horizon]["f1"]
        metrics[horizon]["best_threshold"] = metrics_best[horizon]["threshold"]

    # Optional: q90/q80 metrics for reference
    extra_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label_name in ("q90", "q80"):
        y_true = _build_binary_labels(dataloader.dataset, label_name)
        if y_true is None or y_true.shape[0] != preds.shape[0]:
            continue
        thresholds_extra = getattr(dataloader.dataset, label_name, None)
        if thresholds_extra is not None:
            thresholds_extra = np.asarray(thresholds_extra, dtype=np.float32)
            if thresholds_extra.shape[0] != preds.shape[1]:
                thresholds_extra = None
        extra_metrics[label_name] = compute_metrics_per_horizon(
            y_true,
            preds,
            threshold=threshold,
            best_threshold=best_threshold,
            thresholds=thresholds_extra,
        )
        extra_best = compute_metrics_per_horizon(
            y_true,
            preds,
            threshold=threshold,
            best_threshold=True,
            thresholds=None,
        )
        for horizon in extra_metrics[label_name]:
            extra_metrics[label_name][horizon]["f1_best"] = extra_best[horizon]["f1"]
            extra_metrics[label_name][horizon]["best_threshold"] = extra_best[horizon]["threshold"]

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, metrics, extra_metrics, _compute_pos_stats(y_true_q95)


def train_cnn(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device: str,
    epochs: int,
    early_stop_patience: int,
    horizon_weights: Optional[List[float]],
    severity_loss_weight: float,
    checkpoint_path: Path,
    log_path: Optional[Path],
    metrics_path: Path,
    threshold: float = 0.5,
    best_threshold: bool = False,
    best_metric: str = "auprc",
    min_epochs: int = 20,
    freeze_backbone_epochs: int = 0,
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, float]]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_state = None
    best_metrics: Dict[str, Dict[str, float]] = {}
    best_val_loss: Optional[float] = None
    best_epoch = -1

    model.to(device)

    def _set_backbone_trainable(trainable: bool) -> None:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return
        for param in backbone.parameters():
            param.requires_grad = trainable

    if freeze_backbone_epochs > 0:
        _set_backbone_trainable(False)

    epochs_without_improve = 0

    if horizon_weights is not None:
        if len(horizon_weights) == 1:
            horizon_weights = horizon_weights * len(HORIZONS)
        elif len(horizon_weights) != len(HORIZONS):
            raise ValueError(
                f"horizon_weights length {len(horizon_weights)} does not match horizons {len(HORIZONS)}"
            )

    log_file = log_path.open("w", encoding="utf-8") if log_path is not None else None
    try:
        for epoch in range(1, epochs + 1):
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
                _set_backbone_trainable(True)

            model.train()
            train_losses: List[float] = []
            train_preds: List[float] = []

            for batch in train_loader:
                images = batch["image"].to(device)
                aux = batch.get("aux")
                if aux is not None:
                    aux = aux.to(device)
                targets = batch["severity"].to(device)

                optimizer.zero_grad()
                outputs = model(images, aux)
                loss = _compute_weighted_loss(
                    outputs,
                    targets,
                    loss_fn,
                    horizon_weights,
                    severity_weight=severity_loss_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend(outputs.detach().cpu().numpy().flatten().tolist())

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            train_pred_std = float(np.std(train_preds)) if train_preds else 0.0

            val_loss, val_metrics, val_extra, val_pos_stats = evaluate_cnn(
                model,
                val_loader,
                loss_fn,
                device,
                horizon_weights=horizon_weights,
                severity_loss_weight=severity_loss_weight,
                threshold=threshold,
                best_threshold=best_threshold,
            )

            summary = summarize_metrics(val_metrics) if val_metrics else {}
            if best_metric == "auprc":
                score = summary.get("mean_auprc", float("nan"))
            elif best_metric == "loss":
                score = -val_loss if np.isfinite(val_loss) else float("nan")
            else:
                score = summary.get(f"mean_{best_metric}", float("nan"))

            if not np.isfinite(score):
                score = -val_loss if np.isfinite(val_loss) else float("-inf")

            improved = False
            if best_state is None:
                improved = True
            elif np.isfinite(score) and score > best_score:
                improved = True

            if improved:
                best_score = score
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                }
                best_metrics = val_metrics
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(best_state, checkpoint_path)
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pred_std": train_pred_std,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "summary": summary,
                "pos_stats": val_pos_stats,
                "extra_metrics": val_extra,
                "best_epoch": best_epoch,
                "best_score": best_score if np.isfinite(best_score) else None,
                "early_stop_counter": epochs_without_improve,
                "early_stop_patience": early_stop_patience,
                "min_epochs": min_epochs,
                "freeze_backbone_epochs": freeze_backbone_epochs,
            }
            if log_file is not None:
                log_file.write(json.dumps(_sanitize_for_json(log_entry), ensure_ascii=False) + "\n")
                log_file.flush()

            mean_auprc = summary.get("mean_auprc", float("nan"))
            print(
                f"Epoch {epoch:03d} | "
                f"Train: {train_loss:.4f} (std={train_pred_std:.4f}) | "
                f"Val: {val_loss:.4f} | "
                f"mAUPRC: {mean_auprc:.4f} | "
                f"Best@{best_epoch} ({best_score:.4f}) | "
                f"NoImprove: {epochs_without_improve}/{early_stop_patience}"
            )

            if epoch < min_epochs:
                continue

            if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {early_stop_patience} epochs after {min_epochs} min epochs)."
                )
                break
    finally:
        if log_file is not None:
            log_file.close()

    if best_state is not None:
        # NOTE: Torch 2.1 + ConvNeXt depthwise conv can throw aliasing errors on load.
        # Clone tensors to break shared storage before loading.
        state_dict = best_state.get("model_state_dict", {})
        safe_state_dict = {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_dict.items()
        }
        model.load_state_dict(safe_state_dict)

    overall = summarize_metrics(best_metrics) if best_metrics else {}
    final_metrics = {
        "model": "cnn",
        "commodity": meta.get("commodity") if meta else None,
        "fold": meta.get("fold") if meta else None,
        "window_size": meta.get("window_size") if meta else None,
        "window": meta.get("window_size") if meta else None,
        "horizons": HORIZONS,
        "overall": overall,
        "summary": overall,
        "per_horizon": best_metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    if best_metrics:
        final_metrics["pos_stats"] = val_pos_stats
    if val_extra:
        final_metrics["extra_metrics"] = val_extra
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(final_metrics), f, ensure_ascii=False, indent=2)

    return best_metrics
