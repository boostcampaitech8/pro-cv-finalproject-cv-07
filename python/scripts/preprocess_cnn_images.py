import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.image_generator import (
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
        "--window_sizes",
        required=True,
        type=int,
        nargs="+",
        help="List of window sizes (e.g., 5 20 60).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if they already exist.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _prepare_output_dirs(base_dir: Path, window_sizes: list[int]) -> dict[int, dict[str, Path]]:
    output_dirs: dict[int, dict[str, Path]] = {}
    for window_size in window_sizes:
        candlestick_dir = base_dir / "candlestick" / f"w{window_size}"
        gaf_dir = base_dir / "GAF" / f"w{window_size}"
        rp_dir = base_dir / "RP" / f"w{window_size}"

        candlestick_dir.mkdir(parents=True, exist_ok=True)
        gaf_dir.mkdir(parents=True, exist_ok=True)
        rp_dir.mkdir(parents=True, exist_ok=True)

        output_dirs[window_size] = {
            "candlestick": candlestick_dir,
            "gaf": gaf_dir,
            "rp": rp_dir,
        }
    return output_dirs


def _resolve_csv_path(commodity: str) -> Path:
    return Path("src/datasets/preprocessing") / f"{commodity}_feature_engineering.csv"


def main() -> None:
    args = parse_args()

    csv_path = _resolve_csv_path(args.commodity)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_columns(df)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Failed to parse some values in 'time' column.")

    df = df.sort_values("time").reset_index(drop=True)

    window_sizes = sorted(set(args.window_sizes))
    base_output = Path("src/datasets/preprocessing") / f"{args.commodity}_cnn_preprocessing"
    output_dirs = _prepare_output_dirs(base_output, window_sizes)

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
            gaf_path = dirs["gaf"] / f"{anchor_date}.png"
            rp_path = dirs["rp"] / f"{anchor_date}.png"

            need_candlestick = args.overwrite or not candlestick_path.exists()
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
