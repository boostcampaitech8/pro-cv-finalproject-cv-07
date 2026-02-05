"""
TFT Interpretation Viewer (FINAL – FIXED)

✔ Feature-level importance (Top-K)
✔ Group-level importance (Price / Technical / News)
✔ Temporal importance from attention
✔ Attention heatmap

Usage:
python scripts/view_interpretation.py \
  --interp_file src/outputs/checkpoints/TFT_corn_fold0_h1-5-10-20/interpretations/interpretation_data.npz \
  --feature_csv src/datasets/preprocessing/corn_feature_engineering.csv \
  --top_k 20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp_file", required=True)
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_dir", default="./interpretation_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load interpretation data
    # -----------------------------------------------------
    data = np.load(args.interp_file, allow_pickle=True)

    if "variable_importance" not in data or "attention_weights" not in data:
        raise ValueError("interpretation_data.npz must contain "
                         "'variable_importance' and 'attention_weights'")

    var_imp = data["variable_importance"]          # (seq_length, num_features)
    attn = data["attention_weights"]               # (seq_length, seq_length)

    print("\nLoaded interpretation data")
    print(f"variable_importance shape: {var_imp.shape}")
    print(f"attention_weights shape: {attn.shape}")

    # -----------------------------------------------------
    # Load feature names
    # -----------------------------------------------------
    df_feat = pd.read_csv(args.feature_csv)
    feature_names = df_feat.drop(columns=["time"], errors="ignore").columns.tolist()

    assert len(feature_names) == var_imp.shape[1], \
        f"Feature mismatch: {len(feature_names)} vs {var_imp.shape[1]}"

    # =====================================================
    # 1️⃣ Feature-level importance
    # =====================================================
    avg_feature_importance = var_imp.mean(axis=0)   # (num_features,)

    top_idx = np.argsort(avg_feature_importance)[-args.top_k:][::-1]

    print("\n" + "=" * 60)
    print("📊 Top-K Feature Importance")
    print("=" * 60)

    for i, idx in enumerate(top_idx, 1):
        print(f"{i:2d}. {feature_names[idx]:<40} | {avg_feature_importance[idx]:.6f}")

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(
        range(len(top_idx)),
        avg_feature_importance[top_idx],
        color="steelblue"
    )
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.xlabel("Importance")
    plt.title("Top Feature Importance (TFT)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()

    # =====================================================
    # 2️⃣ Group-level importance
    # =====================================================
    print("\n" + "=" * 60)
    print("📦 Group-level Importance")
    print("=" * 60)

    def idxs(prefixes):
        return [
            i for i, f in enumerate(feature_names)
            if any(f.startswith(p) for p in prefixes)
        ]

    price_idx = idxs(["open", "high", "low", "close"])
    tech_idx = idxs(["EMA", "log_return", "vol_"])
    news_idx = idxs(["news"])

    total = avg_feature_importance.sum()

    def report(name, idx):
        s = avg_feature_importance[idx].sum() if idx else 0.0
        print(f"{name:<15}: {s:.6f} ({s / total * 100:.2f}%)")

    report("Price", price_idx)
    report("Technical", tech_idx)
    report("News", news_idx)

    # =====================================================
    # 3️⃣ Temporal importance (attention)
    # =====================================================
    temporal_importance = attn.mean(axis=0)  # (seq_length,)
    seq_len = len(temporal_importance)

    top_t = np.argsort(temporal_importance)[::-1][:5]

    print("\n" + "=" * 60)
    print("🕒 Top Time Steps (Attention-based)")
    print("=" * 60)

    for i, t in enumerate(top_t, 1):
        print(f"{i}. t-{seq_len - t - 1} | {temporal_importance[t]:.6f}")

    # Plot temporal importance
    plt.figure(figsize=(10, 4))
    plt.bar(range(seq_len), temporal_importance, color="orange")
    plt.xticks(
        range(seq_len),
        [f"t-{seq_len - i - 1}" for i in range(seq_len)],
        rotation=45
    )
    plt.title("Temporal Importance from Attention")
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_importance.png", dpi=300)
    plt.close()

    # =====================================================
    # 4️⃣ Attention heatmap
    # =====================================================
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        cmap="Blues",
        xticklabels=[f"t-{seq_len - i - 1}" for i in range(seq_len)],
        yticklabels=[f"t-{seq_len - i - 1}" for i in range(seq_len)]
    )
    plt.xlabel("Key timestep")
    plt.ylabel("Query timestep")
    plt.title("Average Attention Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_heatmap.png", dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("✅ Interpretation completed")
    print("=" * 60)
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()