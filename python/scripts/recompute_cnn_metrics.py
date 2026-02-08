import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_cnn import CNNDataset
from src.metrics.metrics_cnn import compute_metrics_per_horizon, summarize_metrics


HORIZONS = [1, 5, 10, 20]


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


def _compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics = compute_metrics_per_horizon(y_true, preds, threshold=0.5, best_threshold=False)
    metrics_best = compute_metrics_per_horizon(y_true, preds, threshold=0.5, best_threshold=True)
    for horizon in metrics:
        metrics[horizon]["f1_best"] = metrics_best[horizon]["f1"]
        metrics[horizon]["best_threshold"] = metrics_best[horizon]["threshold"]
    return metrics


def _load_results(results_dir: Path):
    files = sorted(results_dir.glob("*.json"))
    if not files:
        return [], np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), None
    dates = []
    preds = []
    raw_returns = []
    meta = None
    for path in files:
        data = json.loads(path.read_text())
        if meta is None:
            meta = data.get("meta", {})
        date = data["meta"]["date"]
        sev = data["scores"]["severity"]
        dates.append(date)
        preds.append([sev["h1"], sev["h5"], sev["h10"], sev["h20"]])
        rr = data["scores"].get("raw_returns")
        if rr is None:
            raw_returns.append([np.nan, np.nan, np.nan, np.nan])
        else:
            raw_returns.append([rr["h1"], rr["h5"], rr["h10"], rr["h20"]])
    return dates, np.asarray(preds, dtype=np.float32), np.asarray(raw_returns, dtype=np.float32), meta


def _fallback_returns(dates, dataset: CNNDataset) -> np.ndarray:
    date_to_returns = dict(zip(dataset.anchor_dates, dataset.anchor_returns))
    values = []
    for d in dates:
        if d in date_to_returns:
            values.append(date_to_returns[d])
        else:
            values.append([0.0, 0.0, 0.0, 0.0])
    return np.asarray(values, dtype=np.float32)


def _process_fold(commodity: str, exp_dir: Path, fold_dir: Path) -> Optional[Path]:
    results_dir = fold_dir / "results"
    if not results_dir.exists():
        return None

    dates, preds, raw_returns, meta = _load_results(results_dir)
    if len(dates) == 0:
        return None

    fold_id = int(fold_dir.name.split("_")[-1])
    window_size = int(meta.get("window", 20))
    image_mode = meta.get("image_mode", "candle")

    dataset = CNNDataset(
        commodity=commodity,
        fold=fold_id,
        split="val",
        window_size=window_size,
        image_mode=image_mode,
    )

    if np.isnan(raw_returns).any():
        raw_returns = _fallback_returns(dates, dataset)

    q95 = np.asarray(dataset.q95, dtype=np.float32)
    q90 = np.asarray(dataset.q90, dtype=np.float32)
    q80 = np.asarray(dataset.q80, dtype=np.float32)

    y_q95 = (raw_returns >= q95).astype(int)
    y_q90 = (raw_returns >= q90).astype(int)
    y_q80 = (raw_returns >= q80).astype(int)

    metrics_q95 = _compute_metrics(y_q95, preds)
    extra_metrics = {
        "q90": _compute_metrics(y_q90, preds),
        "q80": _compute_metrics(y_q80, preds),
    }
    pos_stats = _compute_pos_stats(y_q95)

    metrics_path = fold_dir / "metrics.json"
    best_epoch = None
    if metrics_path.exists():
        try:
            best_epoch = json.loads(metrics_path.read_text()).get("best_epoch")
        except json.JSONDecodeError:
            best_epoch = None

    final_metrics = {
        "per_horizon": metrics_q95,
        "summary": summarize_metrics(metrics_q95),
        "best_epoch": best_epoch,
        "pos_stats": pos_stats,
        "extra_metrics": extra_metrics,
    }

    metrics_path.write_text(json.dumps(_sanitize_for_json(final_metrics), ensure_ascii=False, indent=2))
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute CNN metrics.json with q90/q80 and pos_stats.")
    parser.add_argument("--commodity", required=True)
    parser.add_argument("--exp_name", default="", help="Specific experiment name. If empty, process all.")
    args = parser.parse_args()

    root = Path("src/outputs/predictions/cnn") / args.commodity
    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")

    exp_dirs = []
    if args.exp_name:
        exp_dirs = [root / args.exp_name]
    else:
        exp_dirs = [p for p in root.iterdir() if p.is_dir()]

    updated = 0
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        for fold_dir in sorted(exp_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            updated_path = _process_fold(args.commodity, exp_dir, fold_dir)
            if updated_path:
                updated += 1
                print(f"updated: {updated_path}")

    print(f"done. updated {updated} metrics.json file(s).")


if __name__ == "__main__":
    main()
