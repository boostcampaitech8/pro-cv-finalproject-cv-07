from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


HORIZONS = [1, 5, 10, 20]


def _to_numpy(array_like) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.0, 1.0, num=101)
    best_f1 = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_f1, best_thresh


def compute_binary_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    best_threshold: bool = False,
) -> Dict[str, float]:
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)

    auroc = _safe_roc_auc(y_true, y_score)
    auprc = _safe_auprc(y_true, y_score)

    if best_threshold:
        f1, best_thresh = _best_f1_threshold(y_true, y_score)
        threshold_used = best_thresh
    else:
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        threshold_used = threshold

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "threshold": float(threshold_used),
    }


def compute_metrics_per_horizon(
    y_true_high: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    best_threshold: bool = False,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    for idx, horizon in enumerate(HORIZONS):
        metrics[str(horizon)] = compute_binary_metrics(
            y_true_high[:, idx],
            y_score[:, idx],
            threshold=threshold,
            best_threshold=best_threshold,
        )

    return metrics


def summarize_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    auroc_vals = [metrics[str(h)]["auroc"] for h in HORIZONS]
    auprc_vals = [metrics[str(h)]["auprc"] for h in HORIZONS]
    f1_vals = [metrics[str(h)]["f1"] for h in HORIZONS]
    f1_best_vals = [
        metrics[str(h)].get("f1_best")
        for h in HORIZONS
        if metrics[str(h)].get("f1_best") is not None
    ]

    summary = {
        "mean_auroc": float(np.nanmean(auroc_vals)),
        "mean_auprc": float(np.nanmean(auprc_vals)),
        "mean_f1": float(np.nanmean(f1_vals)),
    }
    if f1_best_vals:
        summary["mean_f1_best"] = float(np.nanmean(f1_best_vals))
    return summary
