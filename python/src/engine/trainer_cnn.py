import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.metrics.metrics_cnn import compute_metrics_per_horizon, summarize_metrics


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


def evaluate_cnn(
    model: torch.nn.Module,
    dataloader,
    loss_fn,
    device: str,
    horizon_weights: Optional[List[float]] = None,
    severity_loss_weight: float = 0.0,
    threshold: float = 0.5,
    best_threshold: bool = False,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
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

    raw_returns = _collect_raw_returns(dataloader.dataset)
    if raw_returns is None or raw_returns.shape[0] != preds.shape[0]:
        y_true_high = (targets >= 1.0).astype(int)
    else:
        q95 = np.asarray(dataloader.dataset.q95, dtype=np.float32)
        y_true_high = (raw_returns >= q95).astype(int)

    metrics = compute_metrics_per_horizon(
        y_true_high,
        preds,
        threshold=threshold,
        best_threshold=best_threshold,
    )
    metrics_best = compute_metrics_per_horizon(
        y_true_high,
        preds,
        threshold=threshold,
        best_threshold=True,
    )
    for horizon in metrics:
        metrics[horizon]["f1_best"] = metrics_best[horizon]["f1"]
        metrics[horizon]["best_threshold"] = metrics_best[horizon]["threshold"]

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, metrics


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
    log_path: Path,
    metrics_path: Path,
    threshold: float = 0.5,
    best_threshold: bool = False,
    best_metric: str = "auprc",
) -> Dict[str, Dict[str, float]]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_state = None
    best_metrics: Dict[str, Dict[str, float]] = {}
    best_epoch = -1

    model.to(device)

    epochs_without_improve = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: List[float] = []

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

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            val_loss, val_metrics = evaluate_cnn(
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
            score = summary.get("mean_auprc", float("nan"))
            if best_metric == "loss":
                score = -val_loss
            if not np.isfinite(score):
                score = -val_loss

            if best_state is None or score > best_score:
                best_score = score
                best_state = {
                    "model_state_dict": model.state_dict(),
                }
                best_metrics = val_metrics
                best_epoch = epoch
                torch.save(best_state, checkpoint_path)
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "summary": summary,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "early_stop_counter": epochs_without_improve,
                "early_stop_patience": early_stop_patience,
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val mAUPRC: {summary.get('mean_auprc', float('nan')):.4f}"
            )

            if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {early_stop_patience} epochs)."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    final_metrics = {
        "per_horizon": best_metrics,
        "summary": summarize_metrics(best_metrics) if best_metrics else {},
        "best_epoch": best_epoch,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    return best_metrics
