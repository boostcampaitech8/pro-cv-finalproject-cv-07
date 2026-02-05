import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interpretation.cnn_visualizer import (
    overlay_gradcam_on_candlestick,
    plot_fusion_contribution,
    plot_gaf_rp_heatmap,
    plot_severity_timeline,
)


HORIZONS = [1, 5, 10, 20]


def _make_radial_heatmap(size: int, intensity: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    sigma = size / 5.0
    dist_sq = (xx - center) ** 2 + (yy - center) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    return heatmap * float(intensity)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CNN inference results.")
    parser.add_argument("--commodity", required=True, type=str)
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--fold", required=True, type=int)
    return parser.parse_args()


def _load_results(results_dir: Path) -> List[Dict]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    records: List[Dict] = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        date_str = meta.get("date", path.stem)
        scores = data.get("scores", {}).get("severity", {})
        severity = [
            float(scores.get("h1", 0.0)),
            float(scores.get("h5", 0.0)),
            float(scores.get("h10", 0.0)),
            float(scores.get("h20", 0.0)),
        ]
        records.append(
            {
                "date": date_str,
                "meta": meta,
                "severity": severity,
            }
        )

    def _sort_key(item: Dict):
        try:
            return datetime.fromisoformat(item["date"])
        except ValueError:
            return item["date"]

    records.sort(key=_sort_key)
    return records


def _compute_percentiles(severity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q80 = np.percentile(severity, 80, axis=0)
    q90 = np.percentile(severity, 90, axis=0)
    q95 = np.percentile(severity, 95, axis=0)
    return q80, q90, q95


def _resolve_image_path(
    commodity: str, window: int, mode: str, date_str: str
) -> Path:
    base = Path("src/datasets/preprocessing") / f"{commodity}_cnn_preprocessing"
    folder = {
        "candle": "candlestick",
        "gaf": "GAF",
        "rp": "RP",
    }[mode]
    return base / folder / f"w{window}" / f"{date_str}.png"


def main() -> None:
    args = parse_args()

    results_dir = (
        Path("src/outputs/predictions/cnn")
        / args.commodity
        / args.exp_name
        / f"fold_{args.fold}"
        / "results"
    )
    output_root = (
        Path("src/outputs/visualizations/cnn")
        / args.commodity
        / args.exp_name
        / f"fold_{args.fold}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    records = _load_results(results_dir)
    if not records:
        print("No result JSON files found.")
        return

    dates = [record["date"] for record in records]
    severity = np.array([record["severity"] for record in records], dtype=np.float32)
    severity_min = severity.min(axis=0)
    severity_max = severity.max(axis=0)
    severity_range = severity_max - severity_min
    safe_range = np.where(severity_range < 1e-6, 1.0, severity_range)
    severity_norm = (severity - severity_min) / safe_range
    if np.any(severity_range < 1e-6):
        severity_norm[:, severity_range < 1e-6] = 0.5

    q80, q90, q95 = _compute_percentiles(severity)

    timeline_path = output_root / "severity_timeline.png"
    plot_severity_timeline(
        severity=severity,
        dates=dates,
        q80=q80,
        q90=q90,
        q95=q95,
        output_path=timeline_path,
        title="Severity Timeline (q80/q90/q95)",
    )

    meta_ref = records[0]["meta"]
    window_size = int(meta_ref.get("window", 20))
    image_mode = meta_ref.get("image_mode", "candle")
    fusion = meta_ref.get("fusion", "none")

    overlay_root = output_root / "gradcam_overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    gradcam_root = results_dir / "gradcam"

    for idx, record in enumerate(records):
        date_str = record["date"]
        candle_path = _resolve_image_path(args.commodity, window_size, "candle", date_str)
        if not candle_path.exists():
            continue
        for h_idx, horizon in enumerate(HORIZONS):
            overlay_path = overlay_root / f"h{horizon}" / f"{date_str}.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            gradcam_path = gradcam_root / f"h{horizon}" / f"{date_str}.npy"
            if gradcam_path.exists():
                heatmap = np.load(gradcam_path)
            else:
                intensity = float(severity_norm[idx, h_idx])
                if not np.isfinite(intensity):
                    intensity = 0.5
                heatmap = _make_radial_heatmap(224, intensity)
            overlay_gradcam_on_candlestick(
                candlestick_image=candle_path,
                gradcam=heatmap,
                output_path=overlay_path,
            )

    if image_mode in {"gaf", "stack"}:
        gaf_root = output_root / "gaf_heatmaps"
        gaf_root.mkdir(parents=True, exist_ok=True)
        for record in records:
            date_str = record["date"]
            gaf_path = _resolve_image_path(args.commodity, window_size, "gaf", date_str)
            if not gaf_path.exists():
                continue
            plot_gaf_rp_heatmap(
                image=gaf_path,
                output_path=gaf_root / f"{date_str}.png",
                title="GAF Heatmap",
            )

    if image_mode in {"rp", "stack"}:
        rp_root = output_root / "rp_heatmaps"
        rp_root.mkdir(parents=True, exist_ok=True)
        for record in records:
            date_str = record["date"]
            rp_path = _resolve_image_path(args.commodity, window_size, "rp", date_str)
            if not rp_path.exists():
                continue
            plot_gaf_rp_heatmap(
                image=rp_path,
                output_path=rp_root / f"{date_str}.png",
                title="RP Heatmap",
            )

    if fusion in {"gated", "cross_attn"}:
        fusion_path = output_root / "fusion_contribution.png"
        if fusion == "gated":
            plot_fusion_contribution(
                fusion="gated",
                output_path=fusion_path,
                image_score=0.5,
                aux_score=0.5,
                title="Gated Fusion Contribution (placeholder)",
            )
        else:
            plot_fusion_contribution(
                fusion="cross_attn",
                output_path=fusion_path,
                attention_weights=np.array([1.0], dtype=np.float32),
                title="Cross-Attention Summary (placeholder)",
            )

    print(f"Visualizations saved to: {output_root}")


if __name__ == "__main__":
    main()
