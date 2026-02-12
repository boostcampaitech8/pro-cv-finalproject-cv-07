#!/usr/bin/env python
"""
Publish model outputs to BigQuery and GCS, and generate a combined TFT+DeepAR plot.

Example:
  /data/ephemeral/home/airflow/bin/python scripts/publish_outputs.py \
    --symbol corn \
    --window 20 \
    --tft_predictions outputs/predictions/corn_2025-10-02_tft_eval/tft_predictions.csv \
    --deepar_predictions outputs/predictions/corn_2025-10-02_deepar_eval/deepar_predictions.csv \
    --cnn_predictions outputs/predictions/corn_2025-10-02_cnn_eval/cnn_predictions.csv \
    --data_dir shared/src/datasets/local_bq_like/corn \
    --output_dir outputs/predictions/corn_2025-10-02_bundle \
    --gcs_bucket boostcamp-final-proj \
    --gcs_prefix "predictions/{symbol}/w{window}/{base_date}" \
    --project_id esoteric-buffer-485608-g5 \
    --dataset_id final_proj \
    --table_id predict_price \
    --upload_bq \
    --upload_gcs
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import matplotlib.pyplot as plt

from google.cloud import bigquery, storage
try:
    import mysql.connector
except Exception:  # pragma: no cover - optional dependency for SQL upload
    mysql = None


@dataclass
class PredBlock:
    base_date: str
    df: pd.DataFrame


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


def _load_pred_df(path: Path, require_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in require_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df["predict_date"] = pd.to_datetime(df["predict_date"], errors="coerce")
    df = df.dropna(subset=["predict_date"])
    return df


def _select_prediction_block(df: pd.DataFrame, window: int, as_of: Optional[str]) -> PredBlock:
    out = df.copy()
    if "window" in out.columns:
        out = out[out["window"] == window]
    if "date" not in out.columns:
        raise ValueError("Prediction CSV missing 'date' column for base_date")
    if as_of is None:
        as_of = str(out["date"].max())
    out = out[out["date"] == as_of]
    if out.empty:
        raise ValueError(f"No prediction rows after filtering (window={window}, as_of={as_of}).")
    return PredBlock(base_date=as_of, df=out)


def _resolve_interpretation_dir(tft_predictions: Path, window: int, base_date: str) -> Optional[Path]:
    root = tft_predictions.parent / f"w{window}" / "interpretations" / base_date
    if root.exists():
        return root

    # Fallback search
    base = tft_predictions.parent / f"w{window}" / "interpretations"
    if base.exists():
        for cand in base.rglob("feature_importance.png"):
            if cand.parent.name == base_date:
                return cand.parent
    return None


def _save_tft_deepar_plot(
    *,
    tft_df: pd.DataFrame,
    deepar_df: pd.DataFrame,
    data_dir: Optional[Path],
    history_days: int,
    output_path: Path,
) -> None:
    tft_pred = tft_df.set_index("predict_date")["predicted_close"].astype(float)
    q10 = deepar_df.set_index("predict_date")["predicted_q10"].astype(float)
    q90 = deepar_df.set_index("predict_date")["predicted_q90"].astype(float)

    all_pred_dates = sorted(set(tft_pred.index) | set(q10.index) | set(q90.index))
    forecast_start = min(all_pred_dates)

    actual_series = None
    hist_cut = None
    if data_dir is not None and data_dir.exists():
        test_path = data_dir / "test_price.csv"
        if test_path.exists():
            test_df = _load_price_df(test_path)
            if "close" in test_df.columns:
                actual_series = test_df.set_index("date")["close"].astype(float)

        try:
            hist_df = _load_history(data_dir)
            if "close" in hist_df.columns:
                hist_cut = hist_df[hist_df["date"] < forecast_start].tail(history_days)
        except FileNotFoundError:
            hist_cut = None

    plt.figure(figsize=(12, 5))
    if hist_cut is not None and not hist_cut.empty:
        plt.plot(hist_cut["date"], hist_cut["close"].astype(float), label=f"History ({history_days}d)", color="#777777")

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _make_bq_rows(
    *,
    model: str,
    symbol: str,
    window: int,
    block: PredBlock,
    predicted_close_col: Optional[str],
    predicted_q10_col: Optional[str],
    predicted_q90_col: Optional[str],
    severity_col: Optional[str],
) -> pd.DataFrame:
    df = block.df.copy()
    df["model"] = model
    df["symbol"] = symbol
    df["base_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["window_size"] = window
    df["predict_date"] = pd.to_datetime(df["predict_date"], errors="coerce").dt.date
    df["predicted_close"] = df[predicted_close_col] if predicted_close_col else pd.NA
    df["predicted_q10"] = df[predicted_q10_col] if predicted_q10_col else pd.NA
    df["predicted_q90"] = df[predicted_q90_col] if predicted_q90_col else pd.NA
    df["severity_level"] = df[severity_col] if severity_col else pd.NA
    df["created_at"] = pd.Timestamp.utcnow()
    return df[
        [
            "model",
            "symbol",
            "base_date",
            "window_size",
            "predict_date",
            "predicted_close",
            "predicted_q10",
            "predicted_q90",
            "severity_level",
            "created_at",
        ]
    ]


def _upload_bq(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    truncate_table: bool = False,
) -> None:
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    if truncate_table:
        client.query(f"TRUNCATE TABLE `{table_ref}`").result()
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[
            bigquery.SchemaField("model", "STRING"),
            bigquery.SchemaField("symbol", "STRING"),
            bigquery.SchemaField("base_date", "DATE"),
            bigquery.SchemaField("window_size", "INTEGER"),
            bigquery.SchemaField("predict_date", "DATE"),
            bigquery.SchemaField("predicted_close", "FLOAT"),
            bigquery.SchemaField("predicted_q10", "FLOAT"),
            bigquery.SchemaField("predicted_q90", "FLOAT"),
            bigquery.SchemaField("severity_level", "STRING"),
            bigquery.SchemaField("image_tft_deepar", "STRING"),
            bigquery.SchemaField("image_tft_feature_importance", "STRING"),
            bigquery.SchemaField("image_tft_temporal_importance", "STRING"),
            bigquery.SchemaField("image_tft_temporal_feature_importance", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ],
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"✓ Uploaded {len(df)} rows to {table_ref}")


def _upload_gcs(files: dict[str, Path], bucket: str, prefix: str, project_id: str) -> None:
    client = storage.Client(project=project_id)
    bucket_obj = client.bucket(bucket)
    for name, path in files.items():
        if not path.exists():
            print(f"⚠️  Skip missing file: {path}")
            continue
        blob_path = f"{prefix}/{name}"
        blob = bucket_obj.blob(blob_path)
        blob.upload_from_filename(path)
        print(f"✓ Uploaded gs://{bucket}/{blob_path}")


def _resolve_sql_config(args: argparse.Namespace) -> dict[str, str | int]:
    def _pick(value: Optional[str], env_key: str, default: Optional[str] = None) -> Optional[str]:
        return value or os.getenv(env_key) or default

    return {
        "host": _pick(args.sql_host, "DB_HOST"),
        "user": _pick(args.sql_user, "DB_USER"),
        "password": _pick(args.sql_pass, "DB_PASS"),
        "database": _pick(args.sql_db, "DB_NAME"),
        "port": int(_pick(args.sql_port, "DB_PORT", "3306") or 3306),
    }


def _symbol_to_product_id(symbol: str) -> int:
    mapping = {
        "corn": 1,
        "wheat": 2,
        "soybean": 3,
        "gold": 4,
        "silver": 5,
        "copper": 6,
    }
    if symbol not in mapping:
        raise ValueError(f"Unknown symbol for SQL mapping: {symbol}")
    return mapping[symbol]


def _upload_sql(
    tft_block: PredBlock,
    *,
    symbol: str,
    window: int,
    sql_config: dict[str, str | int],
    truncate_table: bool = False,
) -> None:
    if mysql is None:
        raise ImportError("mysql-connector-python is required for --upload_sql.")

    product_id = _symbol_to_product_id(symbol)
    base_date = pd.to_datetime(tft_block.base_date, errors="coerce").date()
    df = tft_block.df.copy()
    df["predict_date"] = pd.to_datetime(df["predict_date"], errors="coerce").dt.date
    df = df.dropna(subset=["predict_date", "predicted_close"])
    if df.empty:
        print("⚠️  No TFT predictions to upload to SQL.")
        return

    rows = [
        (product_id, base_date, window, row["predict_date"], float(row["predicted_close"]))
        for _, row in df.iterrows()
    ]

    conn = mysql.connector.connect(
        host=sql_config["host"],
        user=sql_config["user"],
        password=sql_config["password"],
        database=sql_config["database"],
        port=sql_config["port"],
    )
    try:
        cur = conn.cursor()
        if truncate_table:
            cur.execute("TRUNCATE TABLE predict_price")
        cur.execute(
            "DELETE FROM predict_price WHERE product_id=%s AND base_date=%s AND window_size=%s",
            (product_id, base_date, window),
        )
        cur.executemany(
            "INSERT INTO predict_price (product_id, base_date, window_size, predict_date, predicted_close) "
            "VALUES (%s, %s, %s, %s, %s)",
            rows,
        )
        conn.commit()
        print(f"✓ Uploaded {len(rows)} rows to SQL predict_price (product_id={product_id})")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Commodity symbol (e.g., corn)")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--as_of", default=None, help="Base date (YYYY-MM-DD). Default: latest in predictions")
    parser.add_argument("--tft_predictions", required=True)
    parser.add_argument("--deepar_predictions", required=True)
    parser.add_argument("--cnn_predictions", required=True)
    parser.add_argument("--data_dir", default=None, help="Data dir for history/test (optional)")
    parser.add_argument("--history_days", type=int, default=60)
    parser.add_argument("--output_dir", required=True, help="Local output directory for generated plots")

    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--table_id", required=True)
    parser.add_argument("--upload_bq", action="store_true")
    parser.add_argument("--truncate_table", action="store_true", help="Truncate BQ table before upload")

    parser.add_argument("--upload_sql", action="store_true")
    parser.add_argument("--sql_host", default=None)
    parser.add_argument("--sql_user", default=None)
    parser.add_argument("--sql_pass", default=None)
    parser.add_argument("--sql_db", default=None)
    parser.add_argument("--sql_port", default=None)
    parser.add_argument("--truncate_sql", action="store_true", help="Truncate SQL predict_price before insert")

    parser.add_argument("--gcs_bucket", default=None)
    parser.add_argument("--gcs_prefix", default="predict-{symbol}-w{window}")
    parser.add_argument(
        "--gcs_name_pattern",
        default="predict-{symbol}-{suffix}.png",
        help="Filename pattern for uploaded images",
    )
    parser.add_argument("--upload_gcs", action="store_true")

    args = parser.parse_args()

    tft_df = _load_pred_df(Path(args.tft_predictions), ["date", "window", "predict_date", "predicted_close"])
    deepar_df = _load_pred_df(
        Path(args.deepar_predictions),
        ["date", "window", "predict_date", "predicted_close", "predicted_q10", "predicted_q90"],
    )
    cnn_df = _load_pred_df(Path(args.cnn_predictions), ["date", "window", "predict_date", "severity_level"])

    tft_block = _select_prediction_block(tft_df, args.window, args.as_of)
    deepar_block = _select_prediction_block(deepar_df, args.window, tft_block.base_date)
    cnn_block = _select_prediction_block(cnn_df, args.window, tft_block.base_date)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Combined TFT+DeepAR plot
    combined_plot = output_dir / f"{args.symbol}_w{args.window}_tft_deepar.png"
    data_dir = Path(args.data_dir) if args.data_dir else None
    _save_tft_deepar_plot(
        tft_df=tft_block.df,
        deepar_df=deepar_block.df,
        data_dir=data_dir,
        history_days=args.history_days,
        output_path=combined_plot,
    )

    # 2) TFT interpretation images
    interp_dir = _resolve_interpretation_dir(Path(args.tft_predictions), args.window, tft_block.base_date)
    interp_files = {}
    if interp_dir is not None:
        interp_files = {
            "tft_feature_importance.png": interp_dir / "feature_importance.png",
            "tft_temporal_importance.png": interp_dir / "temporal_importance.png",
            "tft_temporal_feature_importance.png": interp_dir / "temporal_feature_importance.png",
        }
    else:
        print("⚠️  Interpretation directory not found; skipping importance images.")

    # 3) Upload to BigQuery
    if args.upload_bq:
        image_uri_map = {}
        if args.upload_gcs and args.gcs_bucket:
            gcs_prefix = args.gcs_prefix.format(
                symbol=args.symbol,
                window=args.window,
            )
            image_uri_map = {
                "image_tft_deepar": f"gs://{args.gcs_bucket}/{gcs_prefix}/{args.gcs_name_pattern.format(symbol=args.symbol, suffix='tft_deepar')}",
                "image_tft_feature_importance": f"gs://{args.gcs_bucket}/{gcs_prefix}/{args.gcs_name_pattern.format(symbol=args.symbol, suffix='tft_feature_importance')}",
                "image_tft_temporal_importance": f"gs://{args.gcs_bucket}/{gcs_prefix}/{args.gcs_name_pattern.format(symbol=args.symbol, suffix='tft_temporal_importance')}",
                "image_tft_temporal_feature_importance": f"gs://{args.gcs_bucket}/{gcs_prefix}/{args.gcs_name_pattern.format(symbol=args.symbol, suffix='tft_temporal_feature_importance')}",
            }

        rows = [
            _make_bq_rows(
                model="tft",
                symbol=args.symbol,
                window=args.window,
                block=tft_block,
                predicted_close_col="predicted_close",
                predicted_q10_col=None,
                predicted_q90_col=None,
                severity_col=None,
            ),
            _make_bq_rows(
                model="deepar",
                symbol=args.symbol,
                window=args.window,
                block=deepar_block,
                predicted_close_col="predicted_close",
                predicted_q10_col="predicted_q10",
                predicted_q90_col="predicted_q90",
                severity_col=None,
            ),
            _make_bq_rows(
                model="cnn",
                symbol=args.symbol,
                window=args.window,
                block=cnn_block,
                predicted_close_col=None,
                predicted_q10_col=None,
                predicted_q90_col=None,
                severity_col="severity_level",
            ),
        ]
        out_df = pd.concat(rows, axis=0, ignore_index=True)
        for col, uri in image_uri_map.items():
            out_df[col] = uri
        _upload_bq(out_df, args.project_id, args.dataset_id, args.table_id, truncate_table=args.truncate_table)

    # 3.5) Upload TFT close predictions to SQL
    if args.upload_sql:
        sql_config = _resolve_sql_config(args)
        missing = [k for k, v in sql_config.items() if v in (None, "")]
        if missing:
            raise ValueError(f"SQL config missing: {missing}")
        _upload_sql(
            tft_block,
            symbol=args.symbol,
            window=args.window,
            sql_config=sql_config,
            truncate_table=args.truncate_sql,
        )

    # 4) Upload images to GCS
    if args.upload_gcs:
        if not args.gcs_bucket:
            raise ValueError("--gcs_bucket is required when --upload_gcs is set")
        gcs_prefix = args.gcs_prefix.format(
            symbol=args.symbol,
            window=args.window,
        )
        files = {
            args.gcs_name_pattern.format(symbol=args.symbol, suffix="tft_deepar"): combined_plot
        }
        files.update(interp_files)
        _upload_gcs(files, args.gcs_bucket, gcs_prefix, args.project_id)


if __name__ == "__main__":
    main()
