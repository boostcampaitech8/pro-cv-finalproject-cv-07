import argparse
import importlib.util
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery, storage

MODEL_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOTENV = os.getenv("AIRFLOW_ENV")
if not DEFAULT_DOTENV:
    DEFAULT_DOTENV = str(MODEL_ROOT.parent.parent / "airflow" / ".env")

def _load_generate_candlestick_image():
    module_path = MODEL_ROOT / "src" / "data" / "image_generator.py"
    spec = importlib.util.spec_from_file_location("image_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load image_generator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_candlestick_image


generate_candlestick_image = _load_generate_candlestick_image()

EMAS_LIST: list[list[int]] = [[]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync candle images to GCS.")
    parser.add_argument("--ds", required=True, help="Execution date (YYYY-MM-DD).")
    parser.add_argument("--project_id", default=None)
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--windows", nargs="*", type=int, default=None)
    parser.add_argument("--chart_type", default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument(
        "--dotenv_path",
        default=DEFAULT_DOTENV,
    )
    return parser.parse_args()



def _load_dotenv(dotenv_path: str | None) -> None:
    if dotenv_path and os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=False)


def _resolve_symbols(raw: list[str] | None) -> list[str]:
    if raw:
        return raw
    env_val = os.getenv("CANDLE_SYMBOLS", "ZC=F,ZW=F,ZS=F,GC=F,SI=F,HG=F")
    return [s.strip() for s in env_val.split(",") if s.strip()]


def _resolve_windows(raw: list[int] | None) -> list[int]:
    if raw:
        return raw
    env_val = os.getenv("CANDLE_WINDOWS", "5,20,60")
    return [int(x) for x in env_val.split(",") if x.strip()]


def _load_price_tables(project_id: str, dataset_id: str, symbols: list[str], ds: str) -> pd.DataFrame:
    if not project_id or not dataset_id:
        raise ValueError("project_id and dataset_id are required.")

    client = bigquery.Client(project=project_id)
    target_date = datetime.strptime(ds, "%Y-%m-%d").date()

    symbols_literal = ", ".join([f'"{s}"' for s in symbols])

    cols = [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "EMA_5",
        "EMA_10",
        "EMA_20",
        "EMA_50",
        "EMA_100",
        "EMA_200",
    ]

    query_train = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.train_price`
        WHERE symbol IN UNNEST([{symbols_literal}])
          AND trade_date <= '{target_date}'
    """
    query_infer = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.inference_price`
        WHERE symbol IN UNNEST([{symbols_literal}])
          AND trade_date <= '{target_date}'
    """

    df_train = client.query(query_train).to_dataframe()
    df_infer = client.query(query_infer).to_dataframe()

    df = pd.concat([df_train, df_infer], ignore_index=True)
    if df.empty:
        return df

    df = df[cols]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    df = df.set_index("trade_date")
    return df


def _get_existing_blobs(bucket: storage.Bucket, prefix: str) -> set[str]:
    return set(blob.name for blob in bucket.list_blobs(prefix=prefix))


def _generate_image(
    df_win: pd.DataFrame,
    emas: list[int],
    chart_type: str,
    image_size: int,
) -> BytesIO:
    ohlc = df_win[["open", "high", "low", "close"]].astype("float32")
    image = generate_candlestick_image(
        open_prices=ohlc["open"].to_numpy(),
        high_prices=ohlc["high"].to_numpy(),
        low_prices=ohlc["low"].to_numpy(),
        close_prices=ohlc["close"].to_numpy(),
        image_size=image_size,
    )
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG", compress_level=1)
    img_bytes.seek(0)
    return img_bytes


def sync_candle_images(
    ds: str,
    project_id: str,
    dataset_id: str,
    bucket_name: str,
    symbols: list[str],
    windows: list[int],
    chart_type: str,
    image_size: int,
) -> None:
    df = _load_price_tables(project_id, dataset_id, symbols, ds)
    if df.empty:
        print("No rows found. Skip sync.")
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for symbol, group in df.groupby("symbol"):
        group = group.sort_index()
        idx = group.index

        for window in windows:
            for emas in EMAS_LIST:
                ema_suffix = "ohlc" if not emas else "ema_" + "_".join(map(str, emas))
                prefix = f"{symbol}/window_{window}_ohlc/"
                existing = _get_existing_blobs(bucket, prefix=prefix)
                expected: set[str] = set()

                saved = 0
                skipped = 0
                t0 = time.time()

                if len(idx) < window:
                    extras = existing
                    for blob_name in extras:
                        bucket.blob(blob_name).delete()
                    print(
                        f"[{symbol}] w{window} {ema_suffix}: no enough rows, deleted={len(extras)}"
                    )
                    continue

                # Use a window of length `window` that ENDS at the anchor date.
                for pos in range(window - 1, len(idx)):
                    end = idx[pos]
                    end_str = end.strftime("%Y-%m-%d")
                    blob_name = f"{prefix}{end_str}.png"
                    expected.add(blob_name)

                    if blob_name in existing:
                        skipped += 1
                        continue

                    df_win = group.iloc[pos - window + 1 : pos + 1].copy()
                    df_win[["open", "high", "low", "close"]] = df_win[["open", "high", "low", "close"]].astype(
                        "float64"
                    )

                    img_bytes = _generate_image(
                        df_win=df_win,
                        emas=emas,
                        chart_type=chart_type,
                        image_size=image_size,
                    )
                    bucket.blob(blob_name).upload_from_file(img_bytes, content_type="image/png")
                    saved += 1

                    if saved % 50 == 0:
                        dt = time.time() - t0
                        print(
                            f"[{symbol}] w{window} {ema_suffix}: {saved} saved, {skipped} skipped, elapsed={dt:.1f}s"
                        )

                extras = existing - expected
                for blob_name in extras:
                    bucket.blob(blob_name).delete()

                dt = time.time() - t0
                print(
                    f"[{symbol}] w{window} {ema_suffix}: saved={saved}, skipped={skipped}, deleted={len(extras)}, elapsed={dt:.1f}s"
                )


def main() -> None:
    args = _parse_args()
    _load_dotenv(args.dotenv_path)

    project_id = args.project_id or os.getenv("PRICE_BIGQUERY_PROJECT") or os.getenv("BIGQUERY_PROJECT")
    dataset_id = args.dataset_id or os.getenv("PRICE_BIGQUERY_DATASET") or os.getenv("BIGQUERY_DATASET")
    bucket = (
        args.bucket
        or os.getenv("GCS_AI_BUCKET")
        or os.getenv("GCS_SERVER_BUCKET")
        or os.getenv("CANDLE_BUCKET")
        or "boostcamp-final-proj"
    )
    chart_type = args.chart_type or os.getenv("CANDLE_CHART_TYPE", "candle")
    image_size = args.image_size or int(os.getenv("CANDLE_IMAGE_SIZE", "224"))

    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

    credentials_path = (
        os.getenv("PRICE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    symbols = _resolve_symbols(args.symbols)
    windows = _resolve_windows(args.windows)

    if not project_id or not dataset_id:
        raise ValueError("project_id and dataset_id are required (use .env or flags)")

    sync_candle_images(
        ds=args.ds,
        project_id=project_id,
        dataset_id=dataset_id,
        bucket_name=bucket,
        symbols=symbols,
        windows=windows,
        chart_type=chart_type,
        image_size=image_size,
    )


if __name__ == "__main__":
    main()
