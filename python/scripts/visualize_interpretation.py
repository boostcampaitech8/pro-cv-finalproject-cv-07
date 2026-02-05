"""
TFT Interpretation Viewer (Per-Date / Per-Horizon)
✅ Temporal importance CSV 추가

- h1 / h5 / h10 / h20 자동 순회
- 날짜별(npz) 개별 시각화
- Feature importance (PNG + CSV)
- Temporal importance (PNG + CSV) ← 추가!
- Attention heatmap

Usage:
    python visualize_interpretation.py \\
      --checkpoint_dir src/outputs/checkpoints/TFT_corn_fold0_h1-5-10-20 \\
      --fold 0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def resolve_interp_root(checkpoint_dir: Path, fold: int) -> Path:
    """
    Accept both:
    A) fold_dir: .../checkpoints/TFT_corn_fold0_h1-5-10-20
    B) base_dir: .../checkpoints

    Return:
      .../fold_dir/interpretations/fold{fold}
    """
    checkpoint_dir = checkpoint_dir.resolve()

    # Case A: already fold_dir
    direct = checkpoint_dir / "interpretations" / f"fold{fold}"
    if direct.exists():
        return direct

    # Case B: base checkpoints dir -> search fold_dir candidates
    candidates = sorted(checkpoint_dir.glob(f"TFT_*_fold{fold}_h*"))
    for c in candidates:
        cand = c / "interpretations" / f"fold{fold}"
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"interpretation root not found.\\n"
        f"- tried: {direct}\\n"
        f"- also searched: {checkpoint_dir}/TFT_*_fold{fold}_h*/interpretations/fold{fold}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="fold_dir or base checkpoints dir",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    root = resolve_interp_root(Path(args.checkpoint_dir), args.fold)
    print(f"\\n📂 Interpretation root: {root}")

    # Collect all temporal importance for summary
    all_temporal_importance = []

    # horizons: h1, h5, ...
    for h_dir in sorted(root.glob("h*")):
        horizon = h_dir.name
        print(f"\\n{'='*60}")
        print(f"📈 Processing horizon: {horizon}")
        print(f"{'='*60}")

        npz_files = sorted(h_dir.glob("*.npz"))
        if not npz_files:
            print(f"⚠️  No npz files in {h_dir}")
            continue

        for npz_path in npz_files:
            date = npz_path.stem
            out_dir = h_dir / date
            ensure_dir(out_dir)

            data = load_npz(npz_path)

            has_feature = ("variable_importance" in data) and ("feature_names" in data)
            has_attention = ("attention_matrix" in data)

            # =====================================================
            # 1) Feature importance
            # =====================================================
            if has_feature:
                var_imp = data["variable_importance"].astype(np.float64)  # (F,)
                feature_names = data["feature_names"].tolist()

                basic_imp = float(data.get("basic_importance", 0.0))
                news_imp = float(data.get("news_importance", 0.0))

                total_imp = float(var_imp.sum()) + 1e-12

                df_feat = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": var_imp,
                        "importance_percent": var_imp / total_imp * 100.0,
                    }
                ).sort_values("importance", ascending=False)

                df_feat.to_csv(out_dir / "feature_importance_full.csv", index=False)

                # Top-K plot
                top_k = min(args.top_k, len(df_feat))
                df_top = df_feat.head(top_k)[::-1]

                plt.figure(figsize=(10, 6))
                plt.barh(df_top["feature"], df_top["importance"])
                plt.title(f"Feature Importance ({horizon}, {date})")
                plt.xlabel("Importance")
                plt.tight_layout()
                plt.savefig(out_dir / "feature_importance.png", dpi=300)
                plt.close()

                # Group importance
                df_group = pd.DataFrame(
                    [
                        {
                            "group": "Basic(Price+Tech)",
                            "importance": basic_imp,
                            "ratio_percent": basic_imp / total_imp * 100.0,
                        },
                        {
                            "group": "News",
                            "importance": news_imp,
                            "ratio_percent": news_imp / total_imp * 100.0,
                        },
                    ]
                )
                df_group.to_csv(out_dir / "group_importance.csv", index=False)

            has_feature_ts = ("variable_importance_ts" in data) and ("feature_names" in data)

            if has_feature_ts:
                var_ts = data["variable_importance_ts"].astype(np.float64)  # [T,F]
                feature_names = data["feature_names"].tolist()

                # x축 라벨: 실제 날짜(윈도우)
                if "window_dates" in data:
                    xlabels = data["window_dates"].tolist()  # len=T
                else:
                    T = var_ts.shape[0]
                    xlabels = [f"t-{T - i}" for i in range(T)]

                # Top-K feature 선정(시간 평균 기준)
                feat_mean = var_ts.mean(axis=0)  # [F]
                top_k = min(args.top_k, len(feat_mean))
                top_idx = np.argsort(feat_mean)[-top_k:][::-1]

                heat = var_ts[:, top_idx].T  # [top_k, T]
                ylabels = [feature_names[j] for j in top_idx]

                df_heat = pd.DataFrame(heat, index=ylabels, columns=xlabels)
                df_heat.to_csv(out_dir / "temporal_feature_importance_topk.csv")

                plt.figure(figsize=(max(10, len(xlabels)*0.5), max(6, top_k*0.35)))
                sns.heatmap(df_heat, cmap="Blues")
                plt.title(f"Temporal Feature Importance Top-{top_k} ({horizon}, {date})")
                plt.xlabel("Window date")
                plt.ylabel("Feature")
                plt.tight_layout()
                plt.savefig(out_dir / "temporal_feature_importance_heatmap.png", dpi=300)
                plt.close()

            # =====================================================
            # 2) Temporal importance + Attention heatmap
            # =====================================================
            if has_attention:
                attn = data["attention_matrix"].astype(np.float64)  # (T, T)

                if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
                    print(f"⚠️  Bad attention_matrix shape: {attn.shape} @ {npz_path.name}")
                else:
                    T = attn.shape[0]

                    # Label: t-20 ... t-1
                    if "window_dates" in data:
                        labels = data["window_dates"].tolist()
                    else:
                        labels = [f"t-{T - i}" for i in range(T)]

                    # last query row = (t-1 query) attends to keys
                    if "temporal_importance" in data:
                        temporal_imp = data["temporal_importance"].astype(np.float64)  # (T,)
                    else:
                        temporal_imp = attn[-1]  # fallback: last query row

                    # ✅ Temporal importance PNG
                    plt.figure(figsize=(10, 4))
                    plt.bar(range(T), temporal_imp)
                    plt.xticks(range(T), labels, rotation=45)
                    plt.title(f"Temporal Importance (query=t-1) ({horizon}, {date})")
                    plt.ylabel("Attention Weight")
                    plt.tight_layout()
                    plt.savefig(out_dir / "temporal_importance.png", dpi=300)
                    plt.close()

                    # ✅ Temporal importance CSV 저장
                    df_temporal = pd.DataFrame({
                        "timestep": labels,
                        "temporal_importance": temporal_imp
                    })
                    df_temporal.to_csv(out_dir / "temporal_importance.csv", index=False)

                    # Collect for summary
                    all_temporal_importance.append({
                        "horizon": horizon,
                        "date": date,
                        "timesteps": labels,
                        "importances": temporal_imp.tolist()
                    })

                    # Attention heatmap
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(
                        attn,
                        cmap="Blues",
                        xticklabels=labels,
                        yticklabels=labels,
                    )
                    plt.xlabel("Key timestep")
                    plt.ylabel("Query timestep")
                    plt.title(f"Attention Heatmap ({horizon}, {date})")
                    plt.tight_layout()
                    plt.savefig(out_dir / "attention_heatmap.png", dpi=300)
                    plt.close()

            print(f"✓ Saved interpretation for {horizon} / {date}")

    # =====================================================
    # 3) Summary CSV: All temporal importance
    # =====================================================
    if all_temporal_importance:
        summary_path = root / "temporal_importance_summary.csv"
        
        summary_rows = []
        for item in all_temporal_importance:
            horizon = item["horizon"]
            date = item["date"]
            for timestep, importance in zip(item["timesteps"], item["importances"]):
                summary_rows.append({
                    "horizon": horizon,
                    "date": date,
                    "timestep": timestep,
                    "temporal_importance": importance
                })
        
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(summary_path, index=False)
        print(f"\\n✓ Saved temporal importance summary: {summary_path}")

    print(f"\\n{'='*60}")
    print("✅ All interpretations completed")
    print(f"{'='*60}\\n")


if __name__ == "__main__":
    main()