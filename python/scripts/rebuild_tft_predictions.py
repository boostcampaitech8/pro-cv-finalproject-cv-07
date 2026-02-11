#!/usr/bin/env python3
"""
Rebuild combined tft_predictions.csv from per-window results JSON files.

This script is intentionally defensive against different JSON shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _norm_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    # pandas Timestamp / datetime-like / string
    text = str(value)
    return text[:10]


def _get_series(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[List[Any]]:
    for key in keys:
        if key in obj:
            return _as_list(obj[key])
    # Quantiles may be nested
    q = obj.get("quantiles") or obj.get("quantile") or obj.get("predicted_quantiles")
    if isinstance(q, dict):
        for key in keys:
            if key in q:
                return _as_list(q[key])
        # common numeric keys
        if "0.1" in q and "q10" in keys:
            return _as_list(q["0.1"])
        if "0.9" in q and "q90" in keys:
            return _as_list(q["0.9"])
    return None


def _rows_from_series(dates: List[Any],
                      pred: Optional[List[Any]],
                      q10: Optional[List[Any]],
                      q90: Optional[List[Any]]) -> List[Dict[str, Any]]:
    n = len(dates)
    def _pad(lst: Optional[List[Any]]) -> List[Any]:
        if lst is None:
            return [None] * n
        if len(lst) < n:
            return lst + [None] * (n - len(lst))
        return lst[:n]
    pred = _pad(pred)
    q10 = _pad(q10)
    q90 = _pad(q90)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "predict_date": _norm_date(dates[i]),
                "predicted_close": pred[i],
                "predicted_q10": q10[i],
                "predicted_q90": q90[i],
            }
        )
    return rows


def _extract_rows(obj: Any) -> List[Dict[str, Any]]:
    # List of dicts
    if isinstance(obj, list):
        rows: List[Dict[str, Any]] = []
        for item in obj:
            if isinstance(item, dict):
                rows.extend(_extract_rows(item))
        return rows

    if not isinstance(obj, dict):
        return []

    # TFT result schema: prediction_dates {h1:..., ...}, predictions {close:{h1:...}, ...}
    if "prediction_dates" in obj and isinstance(obj.get("prediction_dates"), dict):
        pred_dates = obj["prediction_dates"]
        preds = obj.get("predictions") or {}
        # pick series by common keys
        close = None
        q10 = None
        q90 = None
        if isinstance(preds, dict):
            close = preds.get("close") or preds.get("predicted_close")
            q10 = preds.get("q10") or preds.get("p10") or preds.get("predicted_q10")
            q90 = preds.get("q90") or preds.get("p90") or preds.get("predicted_q90")

        def _ordered(keys_dict: Dict[str, Any]) -> List[Any]:
            # order by h1, h2, ... numeric suffix
            items = sorted(
                keys_dict.items(),
                key=lambda kv: int(kv[0].lstrip("h")) if str(kv[0]).lstrip("h").isdigit() else kv[0],
            )
            return [v for _, v in items]

        dates = _ordered(pred_dates)
        close_list = _ordered(close) if isinstance(close, dict) else _as_list(close)
        q10_list = _ordered(q10) if isinstance(q10, dict) else _as_list(q10)
        q90_list = _ordered(q90) if isinstance(q90, dict) else _as_list(q90)
        return _rows_from_series(dates, close_list, q10_list, q90_list)

    # Nested container keys
    for key in ("predictions", "prediction", "forecast", "results"):
        if key in obj:
            return _extract_rows(obj[key])

    # Direct dict with predict_dates (list or single)
    dates = _as_list(obj.get("predict_dates") or obj.get("predict_date") or obj.get("dates"))
    if dates:
        pred = _get_series(obj, ("predicted_close", "predicted", "pred", "mean", "p50", "median", "close"))
        q10 = _get_series(obj, ("predicted_q10", "q10", "p10"))
        q90 = _get_series(obj, ("predicted_q90", "q90", "p90"))
        return _rows_from_series(dates, pred, q10, q90)

    # Horizon-indexed dict (keys like "1", "2", ...)
    if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
        rows: List[Dict[str, Any]] = []
        for _, v in sorted(obj.items(), key=lambda kv: int(kv[0])):
            rows.extend(_extract_rows(v))
        return rows

    # Single-row dict
    if "predict_date" in obj or "date" in obj:
        return [
            {
                "predict_date": _norm_date(obj.get("predict_date") or obj.get("date")),
                "predicted_close": obj.get("predicted_close") or obj.get("predicted") or obj.get("pred"),
                "predicted_q10": obj.get("predicted_q10") or obj.get("q10"),
                "predicted_q90": obj.get("predicted_q90") or obj.get("q90"),
            }
        ]

    return []


def _parse_window(path: Path) -> Optional[int]:
    # expect .../w20/results/2026-02-06.json
    for parent in path.parents:
        name = parent.name
        if name.startswith("w") and name[1:].isdigit():
            return int(name[1:])
    return None


def rebuild(pred_root: Path, output_path: Optional[Path]) -> Path:
    pred_root = Path(pred_root)
    if output_path is None:
        output_path = pred_root / "tft_predictions.csv"

    rows: List[Dict[str, Any]] = []
    for json_path in sorted(pred_root.glob("w*/results/*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue

        base_date = json_path.stem  # fallback
        window = _parse_window(json_path)
        if isinstance(data, dict) and "meta" in data:
            meta = data.get("meta") or {}
            base_date = meta.get("as_of") or base_date
            window = meta.get("window") or window

        extracted = _extract_rows(data)
        for row in extracted:
            rows.append(
                {
                    "date": base_date,
                    "window": window,
                    **row,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Consistent column order
        ordered = ["date", "window", "predict_date", "predicted_close", "predicted_q10", "predicted_q90"]
        for col in ordered:
            if col not in df.columns:
                df[col] = None
        df = df[ordered]
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True, help="Prediction root containing w*/results/*.json")
    parser.add_argument("--output", default=None, help="Output tft_predictions.csv path")
    args = parser.parse_args()

    out = rebuild(Path(args.pred_root), Path(args.output) if args.output else None)
    print(f"✓ Rebuilt: {out}")


if __name__ == "__main__":
    main()
