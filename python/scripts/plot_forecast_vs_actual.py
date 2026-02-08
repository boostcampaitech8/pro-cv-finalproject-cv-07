#!/usr/bin/env python
"""
Plot predicted vs actual close with recent history.

Supports both TFT and DeepAR outputs that contain:
  date, window, predict_date, predicted_close
Optional: predicted_q10, predicted_q90

Example:
python scripts/plot_forecast_vs_actual.py \
  --predictions src/outputs/predictions/corn_2025-10-02_tft_eval/tft_predictions.csv \
  --data_dir src/datasets/local_bq_like/corn \
  --window 20 \
  --history_days 60 \
  --output_path src/outputs/predictions/corn_2025-10-02_tft_eval/plots/actual_vs_pred.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def _load_price_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def _load_history(data_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("train_price.csv", "inference_price.csv"):
        p = data_dir / name
        if p.exists():
            frames.append(_load_price_df(p))
    if not frames:
        raise FileNotFoundError(f"No train_price.csv or inference_price.csv under {data_dir}")
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return df


def _select_prediction_block(pred_df: pd.DataFrame, window: Optional[int], as_of: Optional[str]) -> pd.DataFrame:
    df = pred_df.copy()
    if "window" in df.columns and window is not None:
        df = df[df["window"] == window]
    if "date" in df.columns:
        if as_of is None:
            as_of = df["date"].max()
        df = df[df["date"] == as_of]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV (TFT or DeepAR)")
    parser.add_argument("--data_dir", required=True, help="Directory containing train_price.csv/inference_price.csv/test_price.csv")
    parser.add_argument("--window", type=int, default=20, help="Window length to filter predictions")
    parser.add_argument("--as_of", default=None, help="As-of date (YYYY-MM-DD). Default: latest in predictions")
    parser.add_argument("--history_days", type=int, default=60, help="Number of past days to show")
    parser.add_argument("--output_path", default=None, help="Output image path (png)")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    data_dir = Path(args.data_dir)
    test_path = data_dir / "test_price.csv"

    pred_df = pd.read_csv(pred_path)
    if "predict_date" not in pred_df.columns or "predicted_close" not in pred_df.columns:
        raise ValueError("predictions CSV must include 'predict_date' and 'predicted_close' columns")

    pred_df["predict_date"] = pd.to_datetime(pred_df["predict_date"], errors="coerce")
    pred_df = pred_df.dropna(subset=["predict_date"])
    pred_df["date"] = pred_df.get("date", None)

    selected = _select_prediction_block(pred_df, args.window, args.as_of)
    if selected.empty:
        raise ValueError("No prediction rows after filtering (check window/as_of).")

    selected = selected.sort_values("predict_date")
    pred_series = selected.set_index("predict_date")["predicted_close"].astype(float)

    q10_series = None
    q90_series = None
    if "predicted_q10" in selected.columns and "predicted_q90" in selected.columns:
        q10_series = selected.set_index("predict_date")["predicted_q10"].astype(float)
        q90_series = selected.set_index("predict_date")["predicted_q90"].astype(float)

    # Load actuals (test_price)
    actual_series = None
    if test_path.exists():
        test_df = _load_price_df(test_path)
        if "close" in test_df.columns:
            actual_series = test_df.set_index("date")["close"].astype(float)

    # Load history (train + inference)
    hist_df = _load_history(data_dir)
    if "close" not in hist_df.columns:
        raise ValueError("close column missing from history data")

    forecast_start = pred_series.index.min()
    hist_cut = hist_df[hist_df["date"] < forecast_start]
    hist_cut = hist_cut.tail(args.history_days)

    plt.figure(figsize=(12, 5))
    plt.plot(hist_cut["date"], hist_cut["close"].astype(float), label=f"History ({args.history_days}d)", color="#777777")

    # Actuals for predicted dates
    if actual_series is not None:
        actual_aligned = actual_series.reindex(pred_series.index)
        plt.plot(actual_aligned.index, actual_aligned.values, label="Actual", color="#1f77b4")

    # Predictions
    plt.plot(pred_series.index, pred_series.values, label="Predicted", color="#ff7f0e")
    if q10_series is not None and q90_series is not None:
        plt.fill_between(q10_series.index, q10_series.values, q90_series.values, color="#ff7f0e", alpha=0.2, label="q10-q90")

    plt.axvline(forecast_start, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.title("Actual vs Predicted Close")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
