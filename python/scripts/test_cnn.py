import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_cnn import CNNDataset, cnn_collate_fn
from src.data.bigquery_loader import load_news_features_bq
from src.engine.trainer_cnn import evaluate_cnn
from src.metrics.metrics_cnn import HORIZONS, summarize_metrics
from src.models.CNN import CNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CNN anomaly model.")

    parser.add_argument("--target_commodity", required=True, type=str)
    parser.add_argument("--fold", required=True, type=int)

    parser.add_argument("--window_size", choices=[5, 20, 60], type=int, required=True)
    parser.add_argument("--image_mode", choices=["candle", "gaf", "rp", "stack", "candle_gaf", "candle_rp"], required=True)
    parser.add_argument("--backbone", choices=["convnext_tiny", "resnet50", "vit_small"], required=True)

    parser.add_argument("--use_aux", action="store_true")
    parser.add_argument("--aux_type", choices=["news"], default="news")
    parser.add_argument("--fusion", choices=["none", "late", "gated", "cross_attn"], default="none")

    parser.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--metrics_path", type=str, default="")
    parser.add_argument(
        "--news_source",
        choices=["csv", "bigquery"],
        default="bigquery",
        help="News feature source (default: bigquery).",
    )
    parser.add_argument("--bq_news_project_id", type=str, default="gcp-practice-484218")
    parser.add_argument("--bq_news_dataset_id", type=str, default="news_data")
    parser.add_argument("--bq_news_table", type=str, default="daily_summary")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    news_df = None
    if args.use_aux and args.aux_type == "news" and args.news_source == "bigquery":
        news_df = load_news_features_bq(
            project_id=args.bq_news_project_id,
            dataset_id=args.bq_news_dataset_id,
            table=args.bq_news_table,
            commodity=args.target_commodity,
        )

    val_ds = CNNDataset(
        commodity=args.target_commodity,
        fold=args.fold,
        split=args.split,
        window_size=args.window_size,
        image_mode=args.image_mode,
        use_aux=args.use_aux,
        aux_type=args.aux_type,
        data_dir=args.data_dir or None,
        split_file=args.split_file or None,
        news_data=news_df,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cnn_collate_fn,
    )

    in_chans = 3 if args.image_mode == "stack" else 2 if args.image_mode in {"candle_gaf", "candle_rp"} else 1
    aux_dim = 0
    if args.use_aux:
        aux_dim += val_ds.news_dim

    num_outputs = len(HORIZONS)
    model = CNN(
        backbone=args.backbone,
        in_chans=in_chans,
        aux_dim=aux_dim,
        fusion=args.fusion,
        dropout=0.1,
        num_outputs=num_outputs,
        pretrained=False,
    )

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        if not args.exp_name:
            raise ValueError("exp_name is required when checkpoint_path is not provided.")
        checkpoint_path = (
            Path("src/outputs/checkpoints/cnn")
            / args.target_commodity
            / args.exp_name
            / f"fold_{args.fold}"
            / "best.pt"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if args.loss == "smooth_l1":
        loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    else:
        loss_fn = torch.nn.MSELoss(reduction="none")

    horizon_weights = [1.0] * num_outputs
    val_loss, metrics, extra_metrics, pos_stats = evaluate_cnn(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
        horizon_weights=horizon_weights,
    )

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
    else:
        metrics_path = (
            Path("src/outputs/predictions/cnn")
            / args.target_commodity
            / (args.exp_name or checkpoint_path.parents[1].name)
            / f"fold_{args.fold}"
            / "metrics.json"
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        f"{args.split}_loss": val_loss,
        "per_horizon": metrics,
        "summary": summarize_metrics(metrics) if metrics else {},
        "pos_stats": pos_stats,
    }
    if extra_metrics:
        summary["extra_metrics"] = extra_metrics
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
