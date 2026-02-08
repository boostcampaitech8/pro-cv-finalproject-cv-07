"""
Compute TFT test metrics from inference outputs + future close prices.

Usage:
python scripts/test_tft_metrics_future.py \
  --inference_root src/outputs/predictions/corn_2025-11-26_tft_eval/w5/results \
  --inference_price src/datasets/local_bq_like/corn/inference_price.csv \
  --future_price src/datasets/corn_future_price.csv \
  --checkpoint_dir src/outputs/checkpoints/corn_2025-11-26_tft_eval/w5 \
  --commodity corn \
  --window 5
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math

import numpy as np
import pandas as pd
import tyro


def _load_close_map(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    if "close" not in df.columns:
        raise ValueError(f"close column missing in {path}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "close"])
    return dict(zip(df["date"].tolist(), df["close"].tolist()))


def _compute_metrics(preds: np.ndarray, trues: np.ndarray, *, include_mape: bool = False) -> Dict[str, float]:
    eps = 1e-8
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - np.mean(trues)) ** 2))
    r2 = float(1 - (ss_res / (ss_tot + eps)))
    da = float(np.mean((preds > 0) == (trues > 0)) * 100.0)
    out = {"MAE": mae, "RMSE": rmse, "R2": r2, "DA": da}
    if include_mape:
        mape = float(np.mean(np.abs((trues - preds) / (np.abs(trues) + eps))) * 100.0)
        out["MAPE"] = mape
    return out


def _interpolate_log_returns(known: Dict[int, float], max_h: int) -> Dict[int, float]:
    if not known:
        return {}
    xs = sorted(known.keys())
    ys = [known[h] for h in xs]
    full: Dict[int, float] = {}
    for h in range(1, max_h + 1):
        if h <= xs[0]:
            full[h] = ys[0]
        elif h >= xs[-1]:
            full[h] = ys[-1]
        else:
            for i in range(1, len(xs)):
                if xs[i] >= h:
                    x0, x1 = xs[i - 1], xs[i]
                    y0, y1 = ys[i - 1], ys[i]
                    t = (h - x0) / (x1 - x0)
                    full[h] = y0 + t * (y1 - y0)
                    break
    return full


def evaluate_from_inference(
    *,
    inference_root: Path,
    inference_price: Path,
    future_price: Path,
    horizons: List[int],
    commodity: str,
    window: int,
    fold: int,
    checkpoint_dir: Path,
    last_only: bool = True,
    interpolate_horizons: bool = False,
    interpolate_max_horizon: int = 20,
) -> Optional[Path]:
    if not inference_root.exists():
        print(f"[WARN] inference_root not found: {inference_root}")
        return None
    if not inference_price.exists():
        print(f"[WARN] inference_price not found: {inference_price}")
        return None

    # Prefer local test_price.csv if present (evaluation-only data)
    test_price = inference_price.parent / "test_price.csv"
    eval_source = "future_price"
    if test_price.exists():
        future_price = test_price
        eval_source = "test_price"
    if not future_price.exists():
        print(f"[WARN] future_price/test_price not found: {future_price}")
        return None

    base_close_map = _load_close_map(inference_price)
    future_close_map = _load_close_map(future_price)
    future_dates_sorted = sorted(future_close_map.keys())

    json_files = sorted(inference_root.glob("*.json"))
    if not json_files:
        print(f"[WARN] No inference JSONs under {inference_root}")
        return None

    samples = []
    for p in json_files:
        payload = json.loads(p.read_text())
        meta = payload.get("meta", {})
        as_of = str(meta.get("as_of", p.stem))[:10]
        preds = payload.get("predictions", {}).get("log_return", {})
        if interpolate_horizons:
            known = {}
            for h in horizons:
                val = preds.get(f"h{h}")
                if val is None:
                    continue
                try:
                    known[h] = float(val)
                except Exception:
                    continue
            full = _interpolate_log_returns(known, interpolate_max_horizon)
            preds = {f"h{h}": full.get(h) for h in range(1, interpolate_max_horizon + 1)}
        pred_dates = payload.get("prediction_dates", {})
        samples.append((as_of, preds, pred_dates))

    # Use last as_of only (deploy-like), unless told otherwise.
    if last_only:
        samples.sort(key=lambda x: x[0])
        samples = samples[-1:]

    eval_horizons = list(horizons)
    if interpolate_horizons:
        eval_horizons = list(range(1, interpolate_max_horizon + 1))

    preds_by_h = {h: [] for h in eval_horizons}
    trues_by_h = {h: [] for h in eval_horizons}
    preds_close_by_h = {h: [] for h in eval_horizons}
    trues_close_by_h = {h: [] for h in eval_horizons}
    base_close_by_h = {h: [] for h in eval_horizons}
    used_pred_dates: List[str] = []

    for as_of, preds, pred_dates in samples:
        base_close = base_close_map.get(as_of)
        if base_close is None or not np.isfinite(base_close):
            continue
        # trading-date horizon mapping
        if as_of not in future_dates_sorted:
            # find insertion point in trading calendar
            future_after = [d for d in future_dates_sorted if d > as_of]
        else:
            idx = future_dates_sorted.index(as_of)
            future_after = future_dates_sorted[idx + 1 :]

        for h in eval_horizons:
            if h - 1 >= len(future_after):
                continue
            target_date = future_after[h - 1]
            key = f"h{h}"
            pred_val = preds.get(key)
            if pred_val is None:
                continue
            actual_close = future_close_map.get(target_date)
            if actual_close is None or not np.isfinite(actual_close):
                continue
            true_lr = math.log(actual_close / base_close)
            pred_lr = float(pred_val)
            pred_close = float(base_close * math.exp(pred_lr))
            preds_by_h[h].append(pred_lr)
            trues_by_h[h].append(float(true_lr))
            preds_close_by_h[h].append(pred_close)
            trues_close_by_h[h].append(float(actual_close))
            base_close_by_h[h].append(float(base_close))
            used_pred_dates.append(target_date)

    per_horizon = {}
    per_horizon_close = {}
    for h in eval_horizons:
        if not preds_by_h[h]:
            continue
        pred_arr = np.array(preds_by_h[h], dtype=np.float32)
        true_arr = np.array(trues_by_h[h], dtype=np.float32)
        per_horizon[str(h)] = _compute_metrics(pred_arr, true_arr)
        pred_close_arr = np.array(preds_close_by_h[h], dtype=np.float32)
        true_close_arr = np.array(trues_close_by_h[h], dtype=np.float32)
        close_metrics = _compute_metrics(pred_close_arr, true_close_arr, include_mape=True)
        # Direction on close should be based on change vs base close (same as log_return sign).
        base_arr = np.array(base_close_by_h[h], dtype=np.float32)
        if len(base_arr) == len(pred_close_arr) and len(base_arr) > 0:
            pred_dir = (pred_close_arr - base_arr) > 0
            true_dir = (true_close_arr - base_arr) > 0
            close_metrics["DA"] = float(np.mean(pred_dir == true_dir) * 100.0)
        per_horizon_close[str(h)] = close_metrics

    if not per_horizon:
        print("[WARN] No valid horizon metrics computed (missing future dates?)")
        return None

    summary = {}
    for k in ["MAE", "RMSE", "R2", "DA"]:
        vals = [v[k] for v in per_horizon.values()]
        summary[k] = float(np.mean(vals)) if vals else None
    summary_close = {}
    for k in ["MAE", "RMSE", "MAPE", "R2", "DA"]:
        vals = [v[k] for v in per_horizon_close.values() if k in v]
        summary_close[k] = float(np.mean(vals)) if vals else None

    used_pred_dates = sorted(set(used_pred_dates))
    out = {
        "model": "tft",
        "commodity": commodity,
        "fold": fold,
        "seq_length": window,
        "horizons": eval_horizons,
        "trained_horizons": horizons,
        "interpolated_horizons": bool(interpolate_horizons),
        "evaluation": {
            "source": eval_source,
            "inference_root": str(inference_root),
            "inference_price": str(inference_price),
            "future_price": str(future_price),
            "as_of_mode": "last_only" if last_only else "all",
        },
        "test_range": {
            "start": used_pred_dates[0] if used_pred_dates else None,
            "end": used_pred_dates[-1] if used_pred_dates else None,
            "n": len(used_pred_dates),
        },
        "per_horizon": per_horizon,
        "summary": summary,
        "per_horizon_close": per_horizon_close,
        "summary_close": summary_close,
        "metric_note": "per_horizon uses log_return; per_horizon_close uses close (MAPE on close).",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = checkpoint_dir / "test_metrics.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✓ Saved test metrics (future eval): {out_path}")
    return out_path


@dataclass
class FutureEvalConfig:
    inference_root: str
    inference_price: str
    future_price: str
    horizons: List[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)  # type: ignore
    commodity: str = "corn"
    window: int = 20
    fold: int = 0
    checkpoint_dir: str = ""
    last_only: bool = True
    interpolate_horizons: bool = False
    interpolate_max_horizon: int = 20


def main(cfg: FutureEvalConfig) -> None:
    checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else Path(cfg.inference_root).parents[1]
    evaluate_from_inference(
        inference_root=Path(cfg.inference_root),
        inference_price=Path(cfg.inference_price),
        future_price=Path(cfg.future_price),
        horizons=list(cfg.horizons),
        commodity=cfg.commodity,
        window=cfg.window,
        fold=cfg.fold,
        checkpoint_dir=checkpoint_dir,
        last_only=cfg.last_only,
        interpolate_horizons=cfg.interpolate_horizons,
        interpolate_max_horizon=cfg.interpolate_max_horizon,
    )


if __name__ == "__main__":
    main(tyro.cli(FutureEvalConfig))
