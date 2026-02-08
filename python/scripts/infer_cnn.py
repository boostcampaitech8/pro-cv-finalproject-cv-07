import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.inference_cnn import run_inference_cnn


def _infer_date_tag(split_path: Path) -> str:
    if not split_path.exists():
        return "latest"
    try:
        import json

        payload = json.loads(split_path.read_text())
        meta = payload.get("meta", {})
        if isinstance(meta, dict):
            inference_window = meta.get("inference_window", {})
            for key in ("end", "end_date"):
                if key in inference_window:
                    return str(inference_window[key])[:10]
            for key in ("test_end_date", "end_date"):
                if key in meta:
                    return str(meta[key])[:10]
    except Exception:
        pass
    return "latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN inference and write JSON outputs.")
    parser.add_argument("--commodity", required=True, type=str)
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--fold", required=True, type=int)

    parser.add_argument("--window_size", choices=[5, 20, 60], type=int, required=True)
    parser.add_argument("--image_mode", choices=["candle", "gaf", "rp", "stack", "candle_gaf", "candle_rp"], required=True)
    parser.add_argument("--backbone", choices=["convnext_tiny", "resnet50", "vit_small"], required=True)
    parser.add_argument(
        "--split",
        choices=["val", "infer"],
        default="infer",
        help="Dataset split to run inference on (default: infer).",
    )

    parser.add_argument("--use_aux", action="store_true")
    parser.add_argument("--aux_type", choices=["news"], default="news")
    parser.add_argument("--fusion", choices=["none", "late", "gated", "cross_attn"], default="none")

    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save_gradcam", action="store_true")
    parser.add_argument(
        "--gradcam_stage",
        type=int,
        default=None,
        help="ConvNeXt stage index for Grad-CAM (e.g., -3 for higher resolution).",
    )
    parser.add_argument(
        "--gradcam_method",
        choices=["gradcam", "layercam"],
        default="gradcam",
        help="Grad-CAM variant to use (default: gradcam).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Data root (e.g., src/datasets/local_bq_like/corn).",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="",
        help="Split JSON filename or path (e.g., corn_split.json).",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="src/outputs/checkpoints/{commodity}_{date}_cnn_eval",
        help="Checkpoint root pattern.",
    )
    parser.add_argument(
        "--prediction_root",
        type=str,
        default="src/outputs/predictions/{commodity}_{date}_cnn_eval",
        help="Prediction root pattern.",
    )
    parser.add_argument(
        "--date_tag",
        type=str,
        default="",
        help="Override date tag used in output paths.",
    )
    parser.add_argument(
        "--all_dates",
        action="store_true",
        help="Export predictions for all available anchor dates (default: latest only).",
    )
    parser.add_argument(
        "--write_json",
        action="store_true",
        help="Also write per-date JSON outputs (default: off).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir or f"src/datasets/local_bq_like/{args.commodity}").resolve()
    split_path = Path(args.split_file) if args.split_file else Path(f"{args.commodity}_split.json")
    if not split_path.is_absolute():
        split_path = (data_dir / split_path).resolve()
    else:
        split_path = split_path.resolve()

    date_tag = args.date_tag or _infer_date_tag(split_path)

    checkpoint_path = args.checkpoint_path
    if not checkpoint_path:
        checkpoint_path = (
            Path(
                args.checkpoint_root.format(
                    commodity=args.commodity,
                    date=date_tag,
                )
            )
            / f"w{args.window_size}"
            / "best_model.pt"
        )

    results_root = run_inference_cnn(
        commodity=args.commodity,
        fold=args.fold,
        split=args.split,
        window_size=args.window_size,
        image_mode=args.image_mode,
        backbone=args.backbone,
        fusion=args.fusion,
        exp_name=args.exp_name,
        use_aux=args.use_aux,
        aux_type=args.aux_type,
        checkpoint_path=str(checkpoint_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device or None,
        save_gradcam=args.save_gradcam,
        gradcam_stage=args.gradcam_stage,
        gradcam_method=args.gradcam_method,
        data_dir=str(data_dir),
        split_file=str(split_path),
        prediction_root=args.prediction_root,
        date_tag=date_tag,
        latest_only=not args.all_dates,
        write_json=args.write_json,
    )

    print(f"Inference JSON saved to: {results_root}")


if __name__ == "__main__":
    main()
