#!/usr/bin/env python
"""
Plot actual vs TFT predicted close and DeepAR q10-q90 band on one chart.

Example:
python scripts/plot_tft_deepar_compare.py \
  --tft_predictions src/outputs/predictions/corn_2025-10-02_tft_eval/tft_predictions.csv \
  --deepar_predictions src/outputs/predictions/corn_2025-10-02_deepar_eval/deepar_predictions.csv \
  --data_dir src/datasets/local_bq_like/corn \
  --window 20 \
  --history_days 60 \
  --output_path src/outputs/predictions/corn_2025-10-02_compare/w20_actual_tft_deepar.png
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


def _load_pred_df(path: Path, require_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in require_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df["predict_date"] = pd.to_datetime(df["predict_date"], errors="coerce")
    df = df.dropna(subset=["predict_date"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tft_predictions", required=True, help="Path to TFT predictions CSV")
    parser.add_argument("--deepar_predictions", required=True, help="Path to DeepAR predictions CSV")
    parser.add_argument("--data_dir", required=True, help="Directory containing train_price.csv/inference_price.csv/test_price.csv")
    parser.add_argument("--window", type=int, default=20, help="Window length to filter predictions")
    parser.add_argument("--as_of", default=None, help="As-of date (YYYY-MM-DD). Default: latest in predictions")
    parser.add_argument("--history_days", type=int, default=60, help="Number of past days to show")
    parser.add_argument("--output_path", default=None, help="Output image path (png)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_path = data_dir / "test_price.csv"

    tft_df = _load_pred_df(
        Path(args.tft_predictions),
        require_cols=["predict_date", "predicted_close"],
    )
    deepar_df = _load_pred_df(
        Path(args.deepar_predictions),
        require_cols=["predict_date", "predicted_q10", "predicted_q90"],
    )

    tft_sel = _select_prediction_block(tft_df, args.window, args.as_of)
    deepar_sel = _select_prediction_block(deepar_df, args.window, args.as_of)
    if tft_sel.empty:
        raise ValueError("No TFT prediction rows after filtering (check window/as_of).")
    if deepar_sel.empty:
        raise ValueError("No DeepAR prediction rows after filtering (check window/as_of).")

    tft_sel = tft_sel.sort_values("predict_date")
    deepar_sel = deepar_sel.sort_values("predict_date")

    tft_pred = tft_sel.set_index("predict_date")["predicted_close"].astype(float)
    q10 = deepar_sel.set_index("predict_date")["predicted_q10"].astype(float)
    q90 = deepar_sel.set_index("predict_date")["predicted_q90"].astype(float)

    # Actuals (test_price)
    actual_series = None
    if test_path.exists():
        test_df = _load_price_df(test_path)
        if "close" in test_df.columns:
            actual_series = test_df.set_index("date")["close"].astype(float)

    # History (train + inference)
    hist_df = _load_history(data_dir)
    if "close" not in hist_df.columns:
        raise ValueError("close column missing from history data")

    all_pred_dates = sorted(set(tft_pred.index) | set(q10.index) | set(q90.index))
    forecast_start = min(all_pred_dates)

    hist_cut = hist_df[hist_df["date"] < forecast_start].tail(args.history_days)

    plt.figure(figsize=(12, 5))
    plt.plot(hist_cut["date"], hist_cut["close"].astype(float), label=f"History ({args.history_days}d)", color="#777777")

    if actual_series is not None:
        actual_aligned = actual_series.reindex(all_pred_dates)
        plt.plot(actual_aligned.index, actual_aligned.values, label="Actual", color="#1f77b4")

    plt.plot(tft_pred.index, tft_pred.values, label="TFT Predicted", color="#ff7f0e")

    plt.fill_between(q10.index, q10.values, q90.values, color="#2ca02c", alpha=0.2, label="DeepAR q10-q90")

    plt.axvline(forecast_start, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.title("Actual vs TFT Predicted & DeepAR q10-q90")
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
