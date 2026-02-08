import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_cnn import CNNDataset, cnn_collate_fn, HORIZONS
from src.engine.trainer_cnn import train_cnn
from src.models.CNN import CNN
from src.utils.set_seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN anomaly model.")

    parser.add_argument("--target_commodity", required=True, type=str)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)

    parser.add_argument("--window_size", choices=[5, 20, 60], type=int, required=True)
    parser.add_argument("--image_mode", choices=["candle", "gaf", "rp", "stack", "candle_gaf", "candle_rp"], required=True)
    parser.add_argument("--backbone", choices=["convnext_tiny", "resnet50", "vit_small"], required=True)

    parser.add_argument("--use_aux", action="store_true")
    parser.add_argument("--aux_type", choices=["news"], default="news")
    parser.add_argument("--fusion", choices=["none", "late", "gated", "cross_attn"], default="none")

    parser.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument(
        "--early_stop_metric",
        choices=["auprc", "loss"],
        default="auprc",
    )
    parser.add_argument("--min_epochs", type=int, default=20)
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Freeze backbone for N initial epochs (head-only warmup).",
    )
    parser.add_argument("--horizon_weights", type=float, nargs="+", default=None)
    parser.add_argument("--severity_loss_weight", type=float, default=1.0)

    parser.add_argument("--exp_name", type=str, default="")
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
        "--date_tag",
        type=str,
        default="",
        help="Override date tag used in output paths.",
    )

    return parser.parse_args()


def build_exp_name(args: argparse.Namespace) -> str:
    if args.exp_name:
        return args.exp_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aux_flag = "aux" if args.use_aux else "noaux"
    return (
        f"{args.backbone}_{args.image_mode}_w{args.window_size}_"
        f"fold{args.fold}_{args.fusion}_{aux_flag}_{timestamp}"
    )


def _infer_date_tag(split_path: Path) -> str:
    if not split_path.exists():
        return datetime.now().strftime("%Y-%m-%d")
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
    return datetime.now().strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    exp_name = build_exp_name(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = args.data_dir or f"src/datasets/local_bq_like/{args.target_commodity}"
    split_path = Path(args.split_file) if args.split_file else Path(f"{args.target_commodity}_split.json")
    if not split_path.is_absolute():
        split_path = Path(data_dir) / split_path

    date_tag = args.date_tag or _infer_date_tag(split_path)

    train_ds = CNNDataset(
        commodity=args.target_commodity,
        fold=args.fold,
        split="train",
        window_size=args.window_size,
        image_mode=args.image_mode,
        use_aux=args.use_aux,
        aux_type=args.aux_type,
        data_dir=data_dir,
        split_file=str(split_path),
    )
    val_ds = CNNDataset(
        commodity=args.target_commodity,
        fold=args.fold,
        split="val",
        window_size=args.window_size,
        image_mode=args.image_mode,
        use_aux=args.use_aux,
        aux_type=args.aux_type,
        data_dir=data_dir,
        split_file=str(split_path),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cnn_collate_fn,
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
        aux_dim += train_ds.news_dim

    num_outputs = len(HORIZONS)
    horizon_weights = args.horizon_weights
    if horizon_weights is None:
        horizon_weights = [1.0] * num_outputs
    elif len(horizon_weights) == 1:
        horizon_weights = horizon_weights * num_outputs
    elif len(horizon_weights) != num_outputs:
        raise ValueError(
            f"--horizon_weights length {len(horizon_weights)} does not match horizons {num_outputs}"
        )

    model = CNN(
        backbone=args.backbone,
        in_chans=in_chans,
        aux_dim=aux_dim,
        fusion=args.fusion,
        dropout=0.1,
        num_outputs=num_outputs,
        pretrained=True,
    )

    if args.loss == "smooth_l1":
        loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    else:
        loss_fn = torch.nn.MSELoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(
        args.checkpoint_root.format(
            commodity=args.target_commodity,
            date=date_tag,
        )
    ) / f"w{args.window_size}"
    checkpoint_path = checkpoint_dir / "best_model.pt"
    metrics_path = checkpoint_dir / "val_metrics.json"
    log_path = checkpoint_dir / "train_log.jsonl"

    train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        horizon_weights=horizon_weights,
        severity_loss_weight=args.severity_loss_weight,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        metrics_path=metrics_path,
        best_metric=args.early_stop_metric,
        min_epochs=args.min_epochs,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        meta={
            "commodity": args.target_commodity,
            "fold": args.fold,
            "window_size": args.window_size,
        },
    )

    print(f"Training complete. Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
