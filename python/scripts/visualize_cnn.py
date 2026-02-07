import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.dataset_cnn import CNNDataset
from src.interpretation.cnn_visualizer import (
    overlay_gradcam_on_candlestick,
    plot_fusion_contribution,
    plot_gaf_rp_heatmap,
    plot_severity_timeline,
)

HORIZONS = [1, 5, 10, 20]

MODE_DIR = {
    "candle": "candlestick",
    "gaf": "GAF",
    "rp": "RP",
    "stack": "candlestick",
}


def _resolve_results_dir(root: Path, exp_name: str, fold: int) -> Path:
    direct = root / exp_name / f"fold_{fold}" / "results"
    if direct.exists():
        return direct
    # Try nested layouts: **/{exp_name}/fold_{fold}/results
    matches = list(root.glob(f"**/{exp_name}/fold_{fold}/results"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"results dir not found for {exp_name} fold {fold}")


def _resolve_output_root(
    vis_root: Path, pred_root: Path, results_dir: Path, exp_name: str, fold: int
) -> Path:
    try:
        rel = results_dir.relative_to(pred_root).parent
        return vis_root / rel
    except Exception:
        return vis_root / exp_name / f"fold_{fold}"


def _load_results(
    results_dir: Path,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[Optional[float]]], Dict]:
    records = []
    for p in results_dir.glob("*.json"):
        records.append((p.name.replace(".json", ""), json.loads(p.read_text())))
    records.sort(key=lambda x: x[0])

    dates: List[str] = []
    pred_series = {f"h{h}": [] for h in HORIZONS}
    actual_series = {f"h{h}": [] for h in HORIZONS}

    meta = {}
    for date, payload in records:
        dates.append(date)
        meta = payload.get("meta", meta)
        scores = payload.get("scores", {}).get("severity", {})
        raw_returns = payload.get("scores", {}).get("raw_returns")
        for h in HORIZONS:
            h_key = f"h{h}"
            pred_series[h_key].append(scores.get(h_key))
            if raw_returns is not None:
                actual_series[h_key].append(raw_returns.get(h_key))
            else:
                actual_series[h_key].append(None)

    return dates, pred_series, actual_series, meta


def _fill_actual_from_dataset(
    dates: List[str],
    actual_series: Dict[str, List[Optional[float]]],
    commodity: str,
    fold: int,
    window: int,
    image_mode: str,
    use_aux: bool,
    aux_type: str,
) -> None:
    # If raw_returns were not saved, reconstruct actuals from the dataset.
    if not dates:
        return
    has_any_actual = any(v is not None for v in actual_series["h1"])
    if has_any_actual:
        return
    try:
        ds = CNNDataset(
            commodity=commodity,
            fold=fold,
            split="val",
            window_size=window,
            image_mode=image_mode,
            use_aux=use_aux,
            aux_type=aux_type,
        )
        date_to_returns = {
            ds.anchor_dates[i]: ds.anchor_returns[i] for i in range(len(ds.anchor_dates))
        }
        for i, date in enumerate(dates):
            ret = date_to_returns.get(date)
            if ret is None:
                continue
            for j, h in enumerate(HORIZONS):
                actual_series[f"h{h}"][i] = float(ret[j])
    except Exception:
        # Fall back to pred-only if dataset lookup fails.
        return


def _compute_thresholds(
    commodity: str,
    fold: int,
    window: int,
    image_mode: str,
    use_aux: bool,
    aux_type: str,
    pred_series: Dict[str, List[float]],
    actual_series: Dict[str, List[Optional[float]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q80 = q90 = q95 = None
    try:
        ds = CNNDataset(
            commodity=commodity,
            fold=fold,
            split="val",
            window_size=window,
            image_mode=image_mode,
            use_aux=use_aux,
            aux_type=aux_type,
        )
        q80 = np.asarray(ds.q80, dtype=np.float32)
        q90 = np.asarray(ds.q90, dtype=np.float32)
        q95 = np.asarray(ds.q95, dtype=np.float32)
        return q80, q90, q95
    except Exception:
        pass

    # Fallback: derive thresholds from raw returns if present (abs value).
    has_actual = any(v is not None for v in actual_series["h1"])
    if has_actual:
        abs_actual = [np.abs(np.asarray(actual_series[f"h{h}"], dtype=np.float32)) for h in HORIZONS]
        q80 = np.asarray([np.nanpercentile(v, 80) for v in abs_actual], dtype=np.float32)
        q90 = np.asarray([np.nanpercentile(v, 90) for v in abs_actual], dtype=np.float32)
        q95 = np.asarray([np.nanpercentile(v, 95) for v in abs_actual], dtype=np.float32)
        return q80, q90, q95

    # Last resort: use prediction distribution.
    q80 = np.nanpercentile([pred_series[k] for k in pred_series], 80, axis=1)
    q90 = np.nanpercentile([pred_series[k] for k in pred_series], 90, axis=1)
    q95 = np.nanpercentile([pred_series[k] for k in pred_series], 95, axis=1)
    return np.asarray(q80), np.asarray(q90), np.asarray(q95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    pred_root = Path("src/outputs/predictions/cnn") / args.commodity
    vis_root = Path("src/outputs/visualizations/cnn") / args.commodity

    results_dir = _resolve_results_dir(pred_root, args.exp_name, args.fold)
    out_root = _resolve_output_root(vis_root, pred_root, results_dir, args.exp_name, args.fold)
    out_root.mkdir(parents=True, exist_ok=True)

    dates, pred_series, actual_series, meta = _load_results(results_dir)
    if not dates:
        raise SystemExit(f"No results found in {results_dir}")

    window = int(meta.get("window", 20))
    image_mode = meta.get("image_mode", "candle")
    aux_type = meta.get("aux_type", "none")
    fusion = meta.get("fusion", "none")
    use_aux = aux_type not in ("none", None)

    _fill_actual_from_dataset(
        dates,
        actual_series,
        args.commodity,
        args.fold,
        window,
        image_mode,
        use_aux,
        aux_type,
    )

    q80, q90, q95 = _compute_thresholds(
        args.commodity,
        args.fold,
        window,
        image_mode,
        use_aux,
        aux_type,
        pred_series,
        actual_series,
    )

    # Build [N,4] arrays for plotting (replace None with NaN)
    severity_mat = np.asarray(
        [[pred_series[f"h{h}"][i] for h in HORIZONS] for i in range(len(dates))],
        dtype=np.float32,
    )
    actual_mat = None
    if any(v is not None for v in actual_series["h1"]):
        actual_mat = np.asarray(
            [
                [
                    (actual_series[f"h{h}"][i] if actual_series[f"h{h}"][i] is not None else np.nan)
                    for h in HORIZONS
                ]
                for i in range(len(dates))
            ],
            dtype=np.float32,
        )

    timeline_path = out_root / "severity_timeline.png"
    plot_severity_timeline(
        severity_mat,
        actual=actual_mat,
        dates=dates,
        q80=q80,
        q90=q90,
        q95=q95,
        output_path=timeline_path,
    )
    # Also save a pred-only view so small variations aren't visually flattened
    # by the much larger scale of actual returns.
    pred_only_path = out_root / "severity_timeline_pred_only.png"
    plot_severity_timeline(
        severity_mat,
        actual=None,
        dates=dates,
        q80=q80,
        q90=q90,
        q95=q95,
        output_path=pred_only_path,
        title="Severity Timeline (Pred Only)",
    )

    # Per-date visualizations
    preprocess_root = (
        Path("src/datasets/preprocessing") / f"{args.commodity}_cnn_preprocessing"
    )
    candle_dir = preprocess_root / MODE_DIR["candle"] / f"w{window}"
    if image_mode in MODE_DIR:
        img_dir = preprocess_root / MODE_DIR[image_mode] / f"w{window}"
    else:
        img_dir = candle_dir

    grad_root = results_dir / "gradcam"
    if grad_root.exists():
        for h in HORIZONS:
            h_dir = grad_root / f"h{h}"
            if not h_dir.exists():
                continue
            out_dir = out_root / "gradcam_overlays" / f"h{h}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for date in dates:
                cam_path = h_dir / f"{date}.npy"
                if not cam_path.exists():
                    continue
                img_path = candle_dir / f"{date}.png"
                if not img_path.exists():
                    continue
                overlay_path = out_dir / f"{date}.png"
                overlay_gradcam_on_candlestick(img_path, cam_path, overlay_path)

    # GAF/RP heatmaps
    if image_mode in ("gaf", "rp", "stack"):
        modes = ["gaf", "rp"] if image_mode == "stack" else [image_mode]
        for mode in modes:
            heat_dir = out_root / f"{mode}_heatmaps"
            heat_dir.mkdir(parents=True, exist_ok=True)
            mode_dir = preprocess_root / MODE_DIR[mode] / f"w{window}"
            for date in dates:
                img_path = mode_dir / f"{date}.png"
                if img_path.exists():
                    plot_gaf_rp_heatmap(img_path, heat_dir / f"{date}.png")

    # Fusion contribution (only if data is available in JSON)
    if fusion in ("gated", "cross_attn"):
        contrib = meta.get("fusion_contrib")
        if contrib is not None:
            out_dir = out_root / "fusion_contrib"
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_fusion_contribution(contrib, out_dir / "fusion_contrib.png", fusion=fusion)

    print(f"Visualizations saved to: {out_root}")


if __name__ == "__main__":
    main()
