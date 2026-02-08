from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
from gluonts.torch.model.predictor import PyTorchPredictor

from src.data.dataset import build_multi_item_dataset, lag_features_by_1day
from src.utils.unified_output import build_forecast_payload, write_json


def _load_predictor(path: Path) -> PyTorchPredictor:
    if not path.exists():
        raise FileNotFoundError(f"Predictor not found: {path}")
    return PyTorchPredictor.deserialize(path)


def _resolve_output_root(
    commodity: str,
    exp_name: str,
    fold: int,
    output_root: Optional[str],
) -> Path:
    if output_root is not None:
        return Path(output_root)
    return (
        Path("src/outputs/predictions/unified/deepar")
        / commodity
        / exp_name
        / f"fold_{fold}"
        / "results"
    )


def run_inference_deepar(
    commodity: str,
    fold: int,
    seq_length: int,
    horizons: Sequence[int],
    *,
    exp_name: Optional[str] = None,
    data_dir: str = "src/datasets",
    checkpoint_dir: str = "src/outputs/checkpoints",
    output_root: Optional[str] = None,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    predictor_prefix: str = "deepar_multi_fold{fold}_h{h}",
) -> Path:
    """
    Run DeepAR inference and write a unified JSON output for the latest date.
    Note: assumes separate predictors per horizon, each with prediction_length == horizon.
    """
    exp_name = exp_name or f"deepar_w{seq_length}_h{'-'.join(map(str, horizons))}"

    data_path = Path(data_dir) / "preprocessing" / f"{commodity}_feature_engineering.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    df["item_id"] = commodity
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in data.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in ["time", "item_id", "close"] and not str(c).startswith("log_return_")
    ]

    df = lag_features_by_1day(df, feature_cols, group_col="item_id", time_col="time")
    if "time_idx" not in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
        df["time_idx"] = df["time"]

    as_of = str(df["time"].max())[:10]

    horizon_values: Dict[int, float] = {}
    quantile_values: Dict[float, Dict[int, float]] = {float(q): {} for q in quantiles}

    dfs = {commodity: df}

    for h in horizons:
        predictor_path = Path(checkpoint_dir) / predictor_prefix.format(fold=fold, h=h)
        predictor = _load_predictor(predictor_path)

        dataset = build_multi_item_dataset(dfs, f"log_return_{h}", feature_cols)
        forecast = next(iter(predictor.predict(dataset)))

        horizon_values[int(h)] = float(forecast.mean[-1])

        for q in quantiles:
            quantile_values[float(q)][int(h)] = float(forecast.quantile(q)[-1])

    ordered_preds = [horizon_values[int(h)] for h in horizons]
    quantile_payload = {
        f"q{q:.2f}": [quantile_values[float(q)][int(h)] for h in horizons]
        for q in quantiles
    }

    payload = build_forecast_payload(
        model="deepar",
        commodity=commodity,
        window=seq_length,
        horizons=horizons,
        as_of=as_of,
        fold=fold,
        predictions=ordered_preds,
        quantiles=quantile_payload,
        model_variant=exp_name,
        extra_meta={
            "predictor_prefix": predictor_prefix,
            "note": "DeepAR outputs use the last step of each horizon-specific forecast.",
        },
    )

    output_root_path = _resolve_output_root(commodity, exp_name, fold, output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    output_path = output_root_path / f"{as_of}.json"
    write_json(payload, output_path)

    return output_root_path
