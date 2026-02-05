import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.inference_cnn import run_inference_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN inference and write JSON outputs.")
    parser.add_argument("--commodity", required=True, type=str)
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--fold", required=True, type=int)

    parser.add_argument("--window_size", choices=[5, 20, 60], type=int, required=True)
    parser.add_argument("--image_mode", choices=["candle", "gaf", "rp", "stack"], required=True)
    parser.add_argument("--backbone", choices=["convnext_tiny", "resnet50", "vit_small"], required=True)

    parser.add_argument("--use_aux", action="store_true")
    parser.add_argument("--aux_type", choices=["volume", "news", "both"], default="volume")
    parser.add_argument("--fusion", choices=["none", "late", "gated", "cross_attn"], default="none")

    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save_gradcam", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = run_inference_cnn(
        commodity=args.commodity,
        fold=args.fold,
        window_size=args.window_size,
        image_mode=args.image_mode,
        backbone=args.backbone,
        fusion=args.fusion,
        exp_name=args.exp_name,
        use_aux=args.use_aux,
        aux_type=args.aux_type,
        checkpoint_path=args.checkpoint_path or None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device or None,
        save_gradcam=args.save_gradcam,
    )

    print(f"Inference JSON saved to: {results_root}")


if __name__ == "__main__":
    main()
