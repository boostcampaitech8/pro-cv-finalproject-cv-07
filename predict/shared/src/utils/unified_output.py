from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

SCHEMA_VERSION = "1.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")


def build_prediction_dates(as_of: str, horizons: Sequence[int]) -> Dict[str, str]:
    base = _parse_date(as_of).date()
    return {
        f"h{int(h)}": (base + timedelta(days=int(h))).isoformat()
        for h in horizons
    }


def map_horizon_values(values: Sequence[float], horizons: Sequence[int]) -> Dict[str, float]:
    if len(values) != len(horizons):
        raise ValueError(
            f"Expected {len(horizons)} values for horizons {list(horizons)}, got {len(values)}."
        )
    return {f"h{int(h)}": float(values[i]) for i, h in enumerate(horizons)}


def severity_label(value: float, q80: float, q90: float, q95: float) -> str:
    if value < q80:
        return "normal"
    if value < q90:
        return "low"
    if value < q95:
        return "medium"
    return "high"


def map_severity_levels(
    values: Sequence[float],
    q80: Sequence[float],
    q90: Sequence[float],
    q95: Sequence[float],
    horizons: Sequence[int],
) -> Dict[str, str]:
    if not (len(values) == len(q80) == len(q90) == len(q95) == len(horizons)):
        raise ValueError("Severity mapping inputs must all match horizons length.")
    levels: Dict[str, str] = {}
    for idx, horizon in enumerate(horizons):
        levels[f"h{int(horizon)}"] = severity_label(
            float(values[idx]),
            float(q80[idx]),
            float(q90[idx]),
            float(q95[idx]),
        )
    return levels


def build_meta(
    *,
    model: str,
    commodity: str,
    window: int,
    horizons: Sequence[int],
    as_of: str,
    fold: int,
    task: str,
    model_variant: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "model": model,
        "commodity": commodity,
        "window": int(window),
        "horizons": [int(h) for h in horizons],
        "as_of": str(as_of)[:10],
        "fold": int(fold),
        "task": task,
        "created_at": _utc_now_iso(),
    }
    if model_variant:
        meta["model_variant"] = model_variant
    if extra:
        meta.update(dict(extra))
    return meta


def build_forecast_payload(
    *,
    model: str,
    commodity: str,
    window: int,
    horizons: Sequence[int],
    as_of: str,
    fold: int,
    predictions: Sequence[float],
    targets: Optional[Sequence[float]] = None,
    quantiles: Optional[Mapping[str, Sequence[float]]] = None,
    model_variant: Optional[str] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
    prediction_dates: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": build_meta(
            model=model,
            commodity=commodity,
            window=window,
            horizons=horizons,
            as_of=as_of,
            fold=fold,
            task="forecast",
            model_variant=model_variant,
            extra=extra_meta,
        ),
        "prediction_dates": prediction_dates or build_prediction_dates(as_of, horizons),
        "predictions": {
            "log_return": map_horizon_values(predictions, horizons),
        },
    }

    if quantiles:
        payload["predictions"]["quantiles"] = {
            str(k): map_horizon_values(v, horizons) for k, v in quantiles.items()
        }

    if targets is not None:
        payload["targets"] = {"log_return": map_horizon_values(targets, horizons)}

    return payload


def build_anomaly_payload(
    *,
    model: str,
    commodity: str,
    window: int,
    horizons: Sequence[int],
    as_of: str,
    fold: int,
    scores: Sequence[float],
    q80: Sequence[float],
    q90: Sequence[float],
    q95: Sequence[float],
    raw_returns: Optional[Sequence[float]] = None,
    model_variant: Optional[str] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": build_meta(
            model=model,
            commodity=commodity,
            window=window,
            horizons=horizons,
            as_of=as_of,
            fold=fold,
            task="anomaly",
            model_variant=model_variant,
            extra=extra_meta,
        ),
        "prediction_dates": build_prediction_dates(as_of, horizons),
        "predictions": {
            "volatility_score": map_horizon_values(scores, horizons),
            "severity": map_severity_levels(scores, q80, q90, q95, horizons),
        },
        "thresholds": {
            "q80": map_horizon_values(q80, horizons),
            "q90": map_horizon_values(q90, horizons),
            "q95": map_horizon_values(q95, horizons),
        },
    }

    if raw_returns is not None:
        payload["targets"] = {
            "volatility": map_horizon_values(raw_returns, horizons),
            "severity": map_severity_levels(raw_returns, q80, q90, q95, horizons),
        }

    return payload


def write_json(payload: Mapping[str, Any], path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent))
