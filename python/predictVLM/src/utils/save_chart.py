import argparse
import time
from pathlib import Path

import pandas as pd
import mplfinance as mpf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


def parse_emas(emas_str: str) -> list:
    """
    EMA 문자열을 리스트로 파싱
    예: "0" -> [], "20" -> [20], "5,10,20" -> [5, 10, 20]
    """
    if emas_str == "0":
        return []
    return [int(x.strip()) for x in emas_str.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate candlestick chart images for VLM experiments")

    parser.add_argument(
        "--data-path",
        type=str,
        default="../../data/corn_future_price.csv",
        help="Path to the CSV file containing price data"
    )
    parser.add_argument(
        "--spans",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="EMA spans to calculate (e.g., --spans 5 10 20)"
    )
    parser.add_argument(
        "--end-start",
        type=str,
        default="2025-10-29",
        help="End date of the range (latest date)"
    )
    parser.add_argument(
        "--end-stop",
        type=str,
        default="2024-05-29",
        help="Start date of the range (earliest date)"
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[20],
        help="Window sizes (e.g., --windows 5 20 60)"
    )
    parser.add_argument(
        "--emas",
        type=str,
        nargs="+",
        default=["0"],
        help="EMA combinations to plot. Use '0' for no EMA, comma-separated for multiple. "
             "(e.g., --emas 0 20 5,10,20)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images",
        help="Base output directory for images"
    )
    parser.add_argument(
        "--chart-type",
        type=str,
        default="candle",
        choices=["candle", "ohlc"],
        help="Chart type: candle or ohlc"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Output image size (square, e.g., 448 for 448x448)"
    )

    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess price data"""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.rename(columns={'Volume': 'volume'}, inplace=True)
    df = df.sort_index()
    return df


def add_ema_features(df: pd.DataFrame, spans: list) -> pd.DataFrame:
    """Add EMA columns to dataframe"""
    df = df.copy()
    for span in spans:
        df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    return df


def generate_images(
    df: pd.DataFrame,
    windows: list,
    emas_list: list,  # list of lists, e.g., [[], [20], [5, 10, 20]]
    end_start: str,
    end_stop: str,
    output_dir: str,
    chart_type: str,
    image_size: int
):
    """Generate candlestick chart images"""

    dpi = mpl.rcParams["figure.dpi"]
    figsize = (504 / dpi, 563 / dpi)

    # Get date range
    end_start_ts = pd.Timestamp(end_start)
    end_stop_ts = pd.Timestamp(end_stop)
    ends = df.index[(df.index <= end_start_ts) & (df.index >= end_stop_ts)]
    pos_arr = df.index.get_indexer(ends)

    print(f"Date range: {end_stop} ~ {end_start}")
    print(f"Total target dates: {len(ends)}")

    for w in windows:
        for emas in emas_list:
            saved = 0
            skipped = 0
            t0 = time.time()

            # 폴더명 생성: emas가 비어있으면 ema0, 아니면 ema5_10_20 형식
            if len(emas) == 0:
                ema_suffix = "ema0"
            else:
                ema_suffix = "ema" + "_".join(map(str, emas))

            out_dir = Path(output_dir) / f"window_{w}_{ema_suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # EMA 컬럼 검증
            for ema in emas:
                ema_col = f"EMA_{ema}"
                if ema_col not in df.columns:
                    raise KeyError(f"EMA_{ema} column not found. Available columns: {list(df.columns)}")

            print(f"\nGenerating: window={w}, emas={emas if emas else 'None'}, output={out_dir}")

            # Count valid samples
            valid_count = sum(1 for pos in pos_arr if pos != -1 and (pos - w) >= 0)

            for end, pos in zip(ends, pos_arr):
                # Skip invalid positions
                start = pos - w
                if start < 0:
                    skipped += 1
                    continue

                # Get window data (end not included, previous w entries)
                df_win = df.iloc[start:pos]

                end_str = end.strftime("%Y-%m-%d")
                out_path = out_dir / f"{end_str}.png"

                # Skip if already exists
                if out_path.exists():
                    skipped += 1
                    continue

                # EMA addplot 생성 (여러 개 가능)
                # EMA 기간별 색상: 5일(갈색), 10일(금색), 20일(파란색)
                ema_colors = {
                    5: '#ef5714',   # 갈색
                    10: '#FFD700',  # 금색
                    20: 'blue',     # 파란색
                }
                addplots = []
                for ema in emas:
                    ema_col = f"EMA_{ema}"
                    color = ema_colors.get(ema, '#ef5714')  # 기본값은 갈색
                    addplots.append(mpf.make_addplot(df_win[ema_col], color=color, width=2.0))

                # Plot kwargs
                plot_kwargs = dict(
                    type=chart_type,
                    style="charles",
                    volume=True,
                    figsize=figsize,
                    returnfig=True,
                )
                if addplots:
                    plot_kwargs["addplot"] = addplots

                # Generate and save chart
                fig, _ = mpf.plot(df_win, **plot_kwargs)
                fig.savefig(
                    out_path,
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0,
                    pil_kwargs={"compress_level": 1},
                )
                plt.close(fig)

                # Resize image
                img = Image.open(out_path)
                img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img_resized.save(out_path, compress_level=1)

                saved += 1

                if saved % 50 == 0:
                    dt = time.time() - t0
                    print(f"  {saved}/{valid_count} saved ({saved/valid_count*100:.1f}%) elapsed={dt:.1f}s last={end_str}")

            dt = time.time() - t0
            print(f"  Completed: {saved} saved, {skipped} skipped, elapsed={dt:.1f}s")


def verify_images(output_dir: str, image_size: int):
    """Verify all generated images have correct size"""
    base_dir = Path(output_dir)
    image_files = list(base_dir.rglob("*.png"))

    mismatch_count = 0
    for img_path in image_files:
        img = Image.open(img_path)
        if img.size != (image_size, image_size):
            print(f"Size mismatch: {img_path.name} is {img.size}, expected ({image_size}, {image_size})")
            mismatch_count += 1

    if mismatch_count == 0:
        print(f"\nAll {len(image_files)} images verified: {image_size}x{image_size}")
    else:
        print(f"\n{mismatch_count} images have incorrect size")


def main():
    args = parse_args()

    # EMA 문자열들을 리스트로 변환
    emas_list = [parse_emas(e) for e in args.emas]

    print("=" * 60)
    print("Chart Image Generator")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"EMA spans to calculate: {args.spans}")
    print(f"Windows: {args.windows}")
    print(f"EMA combinations to plot: {emas_list}")
    print(f"Date range: {args.end_stop} ~ {args.end_start}")
    print(f"Output dir: {args.output_dir}")
    print(f"Chart type: {args.chart_type}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data(args.data_path)
    print(f"Loaded {len(df)} rows")

    # Add EMA features
    print(f"Adding EMA features: {args.spans}")
    df = add_ema_features(df, args.spans)
    print(f"Columns: {list(df.columns)}")

    # Generate images
    generate_images(
        df=df,
        windows=args.windows,
        emas_list=emas_list,
        end_start=args.end_start,
        end_stop=args.end_stop,
        output_dir=args.output_dir,
        chart_type=args.chart_type,
        image_size=args.image_size
    )

    # Verify images
    print("\nVerifying images...")
    verify_images(args.output_dir, args.image_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
