# test_deepar.py
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import tyro
import pandas as pd
import torch
from pathlib import Path
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.torch.model.deepar import DeepAREstimator

from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset import build_multi_item_dataset, lag_features_by_1day


def load_predictor(model_path):
    """저장된 DeepAR predictor 불러오기"""
    from gluonts.torch.model.predictor import PyTorchPredictor
    
    predictor = PyTorchPredictor.deserialize(Path(model_path))
    print(f"✅ Model loaded from: {model_path}")
    return predictor


def predict_and_evaluate(predictor, test_ds, item_ids, save_dir):
    """
    전체 데이터에 대해 예측 + 평가
    """
    # ===============================
    # 1️⃣ 예측 생성
    # ===============================
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    # ===============================
    # 2️⃣ 전체(ALL) 성능 평가
    # ===============================
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

    print("\n========== Overall Test Metrics ==========")
    for k, v in agg_metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:20s}: {v:.4f}")

    # ===============================
    # 3️⃣ Item별 성능
    # ===============================
    item_metrics_df = item_metrics.reset_index()
    item_metrics_df.rename(columns={"index": "item_id"}, inplace=True, errors="ignore")
    
    print("\n========== Item-wise Metrics ==========")
    print(item_metrics_df)

    # ===============================
    # 4️⃣ 결과 저장
    # ===============================
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # aggregate
    agg_df = pd.DataFrame([{"item": "ALL", **agg_metrics}])
    agg_csv = save_dir / "test_aggregate_metrics.csv"
    agg_df.to_csv(agg_csv, index=False, float_format="%.6f")
    print(f"\n✅ Aggregate metrics saved: {agg_csv}")

    # item-wise
    item_csv = save_dir / "test_item_metrics.csv"
    item_metrics_df.to_csv(item_csv, index=False, float_format="%.6f")
    print(f"✅ Item metrics saved: {item_csv}")

    # ===============================
    # 5️⃣ 시각화 (선택)
    # ===============================
    from src.utils.visualization import plot_predictions
    
    plot_predictions(
        forecasts=forecasts,
        tss=tss,
        item_ids=item_ids,
        num_plots=len(item_ids),  # 전부 그리기
        save_dir=save_dir,
    )
    print(f"✅ Prediction plots saved to: {save_dir}")

    return agg_metrics, item_metrics_df


def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    # ===============================
    # 1️⃣ 전체 데이터 로드
    # ===============================
    dfs = {}
    for name in ["corn", "wheat", "soybean"]:
        data_path = os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv")
        data = pd.read_csv(data_path)
        data["item_id"] = name
        data["time"] = pd.to_datetime(data["time"])
        dfs[name] = data

    # ===============================
    # 2️⃣ Feature 추출 + Lag 적용
    # ===============================
    feature_cols = [
        c for c in pd.concat(dfs.values(), ignore_index=True).columns
        if c not in ["time", "item_id", "close"] and not c.startswith("log_return_")
    ]

    for name in list(dfs.keys()):
        dfs[name] = lag_features_by_1day(dfs[name], feature_cols, group_col="item_id", time_col="time")

    # ===============================
    # 3️⃣ Horizon별 테스트
    # ===============================
    all_results = []

    for h in cfg.horizons:
        print(f"\n{'='*60}")
        print(f"🔍 Testing Horizon = {h}")
        print(f"{'='*60}")

        # ===== 데이터셋 생성 =====
        test_ds = build_multi_item_dataset(
            dfs, f"log_return_{h}", feature_cols
        )

        # ===== 모델 불러오기 =====
        # 학습된 모델 경로 (fold=0 기준)
        model_path = Path(cfg.checkpoint_dir) / f"deepar_full_h{h}"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            continue

        predictor = load_predictor(model_path)

        # ===== 예측 + 평가 =====
        save_dir = Path(cfg.checkpoint_dir) / f"test_results_h{h}"
        
        agg_metrics, item_metrics = predict_and_evaluate(
            predictor=predictor,
            test_ds=test_ds,
            item_ids=list(dfs.keys()) ,
            save_dir=save_dir,
        )

        # ===== 결과 수집 =====
        all_results.append({
            "horizon": h,
            "item": "ALL",
            **agg_metrics
        })

        for _, row in item_metrics.iterrows():
            all_results.append({
                "horizon": h,
                "item": row.get("item_id", row.name),
                **row.to_dict()
            })

    # ===============================
    # 4️⃣ 전체 결과 저장
    # ===============================
    results_df = pd.DataFrame(all_results)
    
    final_csv = Path(cfg.checkpoint_dir) / "test_all_horizons_summary.csv"
    results_df.to_csv(final_csv, index=False, float_format="%.6f")
    
    print(f"\n{'='*60}")
    print(f"✅ All test results saved to: {final_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)
