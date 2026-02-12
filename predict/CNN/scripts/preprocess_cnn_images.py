import argparse
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
PREDICT_ROOT = MODEL_ROOT.parent
MODEL_SRC = MODEL_ROOT / "src"
for _path in (MODEL_SRC, PREDICT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_ROOT = PREDICT_ROOT.parent

from data.image_generator import (
    generate_candlestick_image,
    generate_gaf_image,
    generate_rp_image,
)


REQUIRED_COLUMNS = {"time", "open", "high", "low", "close"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CSV into CNN-ready candlestick, GAF, and RP images.",
    )
    parser.add_argument(
        "--commodity",
        required=True,
        type=str,
        help="Commodity name (e.g., corn).",
    )
    parser.add_argument(
        "--seq_lengths",
        required=True,
        type=int,
        nargs="+",
        help="List of sequence lengths (e.g., 5 20 60).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Root data directory (e.g., shared/src/datasets/local_bq_like/corn).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if they already exist.",
    )
    parser.add_argument(
        "--save_gaf_rp",
        action="store_true",
        help="Also save GAF/RP images (default: candlestick only).",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _prepare_output_dirs(
    candle_dir: Path,
    window_sizes: list[int],
    save_gaf_rp: bool,
    gaf_dir: Optional[Path] = None,
    rp_dir: Optional[Path] = None,
) -> dict[int, dict[str, Path]]:
    output_dirs: dict[int, dict[str, Path]] = {}
    for window_size in window_sizes:
        candlestick_dir = candle_dir / f"w{window_size}"

        candlestick_dir.mkdir(parents=True, exist_ok=True)
        gaf_out = None
        rp_out = None
        if save_gaf_rp:
            if gaf_dir is None or rp_dir is None:
                raise ValueError("GAF/RP output dirs must be provided when save_gaf_rp is enabled.")
            gaf_out = gaf_dir / f"w{window_size}"
            rp_out = rp_dir / f"w{window_size}"
            gaf_out.mkdir(parents=True, exist_ok=True)
            rp_out.mkdir(parents=True, exist_ok=True)

        output_dirs[window_size] = {
            "candlestick": candlestick_dir,
            "gaf": gaf_out,
            "rp": rp_out,
        }
    return output_dirs


def _resolve_csv_path(commodity: str, data_dir: str) -> Path:
    if data_dir:
        return Path(data_dir) / "preprocessing" / f"{commodity}_feature_engineering.csv"
    return Path("shared/src/datasets/preprocessing") / f"{commodity}_feature_engineering.csv"


def _load_price_splits(data_dir: str) -> pd.DataFrame:
    base = Path(data_dir) if data_dir else Path("shared/src/datasets")
    candidates = [
        (base / "train_price.csv", "train"),
        (base / "test_price.csv", "test"),
        (base / "inference_price.csv", "inference"),
    ]
    frames = []
    for path, source in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "time" not in df.columns:
            if "date" in df.columns:
                df = df.rename(columns={"date": "time"})
            elif "trade_date" in df.columns:
                df = df.rename(columns={"trade_date": "time"})
        df["_source"] = source
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No price split CSVs found (train/test/inference).")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
    merged = merged.dropna(subset=["time"]).sort_values("time")
    if merged["time"].duplicated().any():
        dup_dates = (
            merged.loc[merged["time"].duplicated(keep=False), "time"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
        sample = ", ".join(dup_dates[:10])
        raise ValueError(
            "Duplicate dates across price splits detected. "
            f"Sample duplicates: {sample}. "
            "Please ensure train/test/inference splits are non-overlapping."
        )
    merged = merged.drop(columns=["_source"], errors="ignore").reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()

    csv_path = _resolve_csv_path(args.commodity, args.data_dir)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = _load_price_splits(args.data_dir)
    _validate_columns(df)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Failed to parse some values in 'time' column.")

    df = df.sort_values("time").reset_index(drop=True)

    window_sizes = sorted(set(args.seq_lengths))
    if args.data_dir:
        candle_output = Path(args.data_dir) / "candle_img"
        gaf_output = Path(args.data_dir) / "gaf_img"
        rp_output = Path(args.data_dir) / "rp_img"
    else:
        base_output = Path("shared/src/datasets/preprocessing") / f"{args.commodity}_cnn_preprocessing"
        candle_output = base_output / "candlestick"
        gaf_output = base_output / "GAF"
        rp_output = base_output / "RP"
    output_dirs = _prepare_output_dirs(
        candle_output,
        window_sizes,
        args.save_gaf_rp,
        gaf_dir=gaf_output if args.save_gaf_rp else None,
        rp_dir=rp_output if args.save_gaf_rp else None,
    )

    total_rows = len(df)
    if total_rows == 0:
        print("No rows found in CSV. Nothing to generate.")
        return

    for window_size in window_sizes:
        if window_size <= 0:
            raise ValueError(f"Window size must be positive. Got {window_size}.")
        if window_size > total_rows:
            print(f"Skipping window size {window_size} (not enough rows).")
            continue

        dirs = output_dirs[window_size]
        start_idx = window_size - 1
        for idx in range(start_idx, total_rows):
            anchor_date = df.loc[idx, "time"].strftime("%Y-%m-%d")

            candlestick_path = dirs["candlestick"] / f"{anchor_date}.png"
            gaf_path = (dirs["gaf"] / f"{anchor_date}.png") if dirs["gaf"] else None
            rp_path = (dirs["rp"] / f"{anchor_date}.png") if dirs["rp"] else None

            need_candlestick = args.overwrite or not candlestick_path.exists()
            need_gaf = False
            need_rp = False
            if args.save_gaf_rp:
                need_gaf = args.overwrite or not gaf_path.exists()
                need_rp = args.overwrite or not rp_path.exists()

            if not (need_candlestick or need_gaf or need_rp):
                continue

            window = df.iloc[idx - window_size + 1 : idx + 1]
            open_prices = window["open"].to_numpy()
            high_prices = window["high"].to_numpy()
            low_prices = window["low"].to_numpy()
            close_prices = window["close"].to_numpy()

            if need_candlestick:
                image = generate_candlestick_image(
                    open_prices,
                    high_prices,
                    low_prices,
                    close_prices,
                    image_size=224,
                )
                image.save(candlestick_path)

            if need_gaf:
                image = generate_gaf_image(close_prices, image_size=224)
                image.save(gaf_path)

            if need_rp:
                image = generate_rp_image(close_prices, image_size=224)
                image.save(rp_path)

        print(f"Completed window size {window_size}.")

    print("Image preprocessing completed.")


if __name__ == "__main__":
    main()
