"""
TFT Inference Script (Unified Output)

Example:
python scripts/inference.py \
  --target_commodity corn \
  --fold 0 \
  --seq_length 20 \
  --horizons 1 5 10 20
"""

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Dict

import tyro

MODEL_ROOT = Path(__file__).resolve().parents[1]
PREDICT_ROOT = MODEL_ROOT.parent
MODEL_SRC = MODEL_ROOT / "src"
for _path in (MODEL_SRC, PREDICT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_ROOT = PREDICT_ROOT.parent

from configs.tft_config import TFTInferenceConfig
from engine.inference_tft import run_inference_tft

def _strip_quotes(value: str) -> str:
    if not value:
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = _strip_quotes(val.strip())
    return env


def _resolve_price_bq_settings(cfg: TFTInferenceConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("PRICE_BIGQUERY_PROJECT") or cfg.bq_project_id,
        "dataset_id": _get("PRICE_BIGQUERY_DATASET") or cfg.bq_dataset_id,
        "credentials_path": _get("PRICE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def _resolve_article_bq_settings(cfg: TFTInferenceConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("ARTICLE_BIGQUERY_PROJECT") or cfg.bq_news_project_id,
        "dataset_id": _get("ARTICLE_BIGQUERY_DATASET") or cfg.bq_news_dataset_id,
        "credentials_path": _get("ARTICLE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "daily_table": _get("TABLE_DAILY_SUMMARY") or cfg.bq_news_table,
    }


def _fetch_latest_inference_date(
    project_id: Optional[str],
    dataset_id: Optional[str],
    table: Optional[str],
    credentials_path: Optional[str] = None,
) -> Optional[str]:
    if not project_id or not dataset_id or not table:
        return None
    try:
        from google.cloud import bigquery
    except Exception:
        return None

    creds = None
    if credentials_path:
        try:
            cred_path = Path(credentials_path)
            if cred_path.is_file():
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(str(cred_path))
        except Exception:
            creds = None

    client = bigquery.Client(project=project_id, credentials=creds)

    def _run_query(field: str) -> Optional[str]:
        query = (
            f"SELECT MAX({field}) AS max_date "
            f"FROM `{project_id}.{dataset_id}.{table}`"
        )
        job = client.query(query)
        rows = list(job.result())
        if not rows:
            return None
        max_date = rows[0].get("max_date")
        if max_date is None:
            return None
        return str(max_date)[:10]

    try:
        date_str = _run_query("trade_date")
    except Exception:
        date_str = None
    if date_str:
        return date_str
    try:
        return _run_query("time")
    except Exception:
        return None


def _infer_pred_root(output_root: Optional[str]) -> Optional[Path]:
    if not output_root:
        return None
    path = Path(output_root)
    if path.name != "results":
        return None
    w_dir = path.parent
    if not w_dir.name.startswith("w"):
        return None
    return w_dir.parent


def _write_combined_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = output_path.read_text().strip()
    else:
        existing = ""
    import pandas as pd

    df_new = pd.DataFrame(rows)
    if df_new.empty:
        return output_path
    if existing:
        df_old = pd.read_csv(output_path)
        if "window_size" in df_old.columns:
            window_vals = df_new["window_size"].dropna().unique().tolist()
            df_old = df_old[~df_old["window_size"].isin(window_vals)]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    ordered = ["base_date", "window_size", "predict_date", "predicted_close"]
    for col in ordered:
        if col not in df.columns:
            df[col] = None
    df = df[ordered]
    df.to_csv(output_path, index=False)
    return output_path


def _run_single(cfg: TFTInferenceConfig) -> None:
    bq_project_id = cfg.bq_project_id
    bq_dataset_id = cfg.bq_dataset_id
    news_project_id = cfg.bq_news_project_id
    news_dataset_id = cfg.bq_news_dataset_id
    news_table = cfg.bq_news_table
    news_credentials_path = None

    if cfg.data_source == "bigquery":
        price_bq = _resolve_price_bq_settings(cfg)
        if price_bq.get("credentials_path") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = price_bq["credentials_path"]
        bq_project_id = price_bq.get("project_id") or bq_project_id
        bq_dataset_id = price_bq.get("dataset_id") or bq_dataset_id

    if cfg.news_source == "bigquery":
        article_bq = _resolve_article_bq_settings(cfg)
        news_project_id = article_bq.get("project_id") or news_project_id
        news_dataset_id = article_bq.get("dataset_id") or news_dataset_id
        news_table = article_bq.get("daily_table") or news_table
        news_credentials_path = article_bq.get("credentials_path")

    exp_name = cfg.exp_name or None
    checkpoint_path = cfg.checkpoint_path or None
    output_root = cfg.output_root or None

    if checkpoint_path is None and cfg.data_source == "bigquery":
        latest_date = _fetch_latest_inference_date(
            project_id=bq_project_id,
            dataset_id=bq_dataset_id,
            table=cfg.bq_inference_table,
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
        if latest_date:
            candidate = (
                Path(cfg.checkpoint_dir)
                / f"{latest_date}_{cfg.target_commodity}_tft"
                / f"w{cfg.seq_length}"
                / "best_model.pt"
            )
            if candidate.exists():
                checkpoint_path = str(candidate)
                print(f"✓ Using checkpoint: {checkpoint_path}")
            else:
                print(f"⚠️  Checkpoint not found: {candidate}")
        else:
            print("⚠️  Could not resolve latest inference date for checkpoint lookup.")

    write_json = not cfg.write_combined_csv
    if cfg.write_combined_csv:
        output_dir, rows = run_inference_tft(
            commodity=cfg.target_commodity,
            fold=cfg.fold,
            seq_length=cfg.seq_length,
            horizons=cfg.horizons,
            exp_name=exp_name,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            data_dir=cfg.data_dir,
            data_source=cfg.data_source,
            bq_project_id=bq_project_id,
            bq_dataset_id=bq_dataset_id,
            bq_train_table=cfg.bq_train_table,
            bq_inference_table=cfg.bq_inference_table,
            news_source=cfg.news_source,
            bq_news_project_id=news_project_id,
            bq_news_dataset_id=news_dataset_id,
            bq_news_table=news_table,
            news_credentials_path=news_credentials_path,
            split=cfg.split,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            device=cfg.device,
            seed=cfg.seed,
            use_variable_selection=cfg.use_variable_selection,
            quantiles=cfg.quantiles,
            include_targets=cfg.include_targets,
            scale_x=cfg.scale_x,
            scale_y=cfg.scale_y,
            save_importance=cfg.save_importance,
            importance_groups=cfg.importance_groups,
            importance_top_k=cfg.importance_top_k,
            save_importance_images=cfg.save_importance_images,
            save_prediction_plot=cfg.save_prediction_plot,
            write_json=write_json,
            return_rows=True,
        )
    else:
        output_dir = run_inference_tft(
        commodity=cfg.target_commodity,
        fold=cfg.fold,
        seq_length=cfg.seq_length,
        horizons=cfg.horizons,
        exp_name=exp_name,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        data_dir=cfg.data_dir,
        data_source=cfg.data_source,
        bq_project_id=bq_project_id,
        bq_dataset_id=bq_dataset_id,
        bq_train_table=cfg.bq_train_table,
        bq_inference_table=cfg.bq_inference_table,
        news_source=cfg.news_source,
        bq_news_project_id=news_project_id,
        bq_news_dataset_id=news_dataset_id,
        bq_news_table=news_table,
        news_credentials_path=news_credentials_path,
        split=cfg.split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
        seed=cfg.seed,
        use_variable_selection=cfg.use_variable_selection,
        quantiles=cfg.quantiles,
        include_targets=cfg.include_targets,
        scale_x=cfg.scale_x,
        scale_y=cfg.scale_y,
        save_importance=cfg.save_importance,
        importance_groups=cfg.importance_groups,
        importance_top_k=cfg.importance_top_k,
        save_importance_images=cfg.save_importance_images,
        save_prediction_plot=cfg.save_prediction_plot,
        write_json=write_json,
    )

    print(f"Unified TFT outputs saved to: {output_dir}")

    if cfg.write_combined_csv:
        pred_root = (
            Path(cfg.combined_csv_root)
            if cfg.combined_csv_root
            else _infer_pred_root(str(output_dir))
        )
        if pred_root is None:
            raise ValueError(
                "Unable to infer combined CSV root. Provide --combined_csv_root "
                "(parent of w*/results)."
            )
        out_csv = _write_combined_csv(rows, pred_root / "tft_predictions.csv")
        print(f"✓ Combined predictions saved: {out_csv}")


def main(cfg: TFTInferenceConfig) -> None:
    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = PREDICT_ROOT / path
        return str(path)

    cfg.data_dir = _resolve_path(cfg.data_dir)
    cfg.checkpoint_dir = _resolve_path(cfg.checkpoint_dir)
    if cfg.output_root:
        cfg.output_root = _resolve_path(cfg.output_root)

    seq_lengths = [int(x) for x in getattr(cfg, "seq_lengths", []) or []]
    if seq_lengths:
        for seq_len in seq_lengths:
            cfg_run = replace(cfg, seq_length=seq_len)
            _run_single(cfg_run)
    else:
        _run_single(cfg)


if __name__ == "__main__":
    main(tyro.cli(TFTInferenceConfig))
