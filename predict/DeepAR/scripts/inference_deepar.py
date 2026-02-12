"""
DeepAR inference (load saved predictor and export quantile CSV/PNG).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import tyro

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.predictor import PyTorchPredictor

MODEL_ROOT = Path(__file__).resolve().parents[1]
PREDICT_ROOT = MODEL_ROOT.parent
MODEL_SRC = MODEL_ROOT / "src"
for _path in (MODEL_SRC, PREDICT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_ROOT = PREDICT_ROOT.parent

from configs.deepar_config import DeepARInferenceConfig
from shared.data.bigquery_loader import load_news_features_bq, load_price_table

# Suppress noisy runtime warnings in batch runs
warnings.filterwarnings("ignore")


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


def _resolve_price_bq_settings(cfg: DeepARInferenceConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("PRICE_BIGQUERY_PROJECT") or cfg.bq_project_id,
        "dataset_id": _get("PRICE_BIGQUERY_DATASET") or cfg.bq_dataset_id,
        "credentials_path": _get("PRICE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    }


def _resolve_article_bq_settings(cfg: DeepARInferenceConfig) -> Dict[str, Optional[str]]:
    env_file = _read_env_file(REPO_ROOT / "airflow" / ".env")

    def _get(key: str) -> Optional[str]:
        return os.getenv(key) or env_file.get(key)

    return {
        "project_id": _get("ARTICLE_BIGQUERY_PROJECT"),
        "dataset_id": _get("ARTICLE_BIGQUERY_DATASET"),
        "credentials_path": _get("ARTICLE_GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "daily_table": _get("TABLE_DAILY_SUMMARY"),
    }


def _fetch_latest_inference_date(
    *,
    data_source: str,
    data_root: Path,
    project_id: Optional[str],
    dataset_id: Optional[str],
    table: Optional[str],
    credentials_path: Optional[str] = None,
) -> Optional[str]:
    if data_source == "bigquery":
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
        query = f"SELECT MAX(trade_date) AS max_date FROM `{project_id}.{dataset_id}.{table}`"
        try:
            rows = list(client.query(query).result())
        except Exception:
            return None
        if not rows:
            return None
        max_date = rows[0].get("max_date")
        if max_date is None:
            return None
        return str(max_date)[:10]

    infer_path = data_root / "inference_price.csv"
    if not infer_path.exists():
        return None
    try:
        df = pd.read_csv(infer_path)
    except Exception:
        return None
    date_col = None
    for cand in ("time", "trade_date", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return None
    max_dt = pd.to_datetime(df[date_col], errors="coerce").max()
    if pd.isna(max_dt):
        return None
    return max_dt.strftime("%Y-%m-%d")


def _load_price_df(
    *,
    data_source: str,
    data_dir: str,
    filename: str,
    commodity: str,
    bq_project_id: str,
    bq_dataset_id: str,
    bq_table: str,
) -> pd.DataFrame:
    if data_source == "bigquery":
        df = load_price_table(
            project_id=bq_project_id,
            dataset_id=bq_dataset_id,
            table=bq_table,
            commodity=commodity,
        )
    else:
        price_path = os.path.join(data_dir, filename)
        df = pd.read_csv(price_path)
        if "time" not in df.columns:
            raise ValueError(f"'time' column missing in {price_path}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["item_id"] = commodity
    return df


def _merge_news(price_df: pd.DataFrame, news_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return price_df
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    news = news_df.copy()
    if "date" not in news.columns and "collect_date" in news.columns:
        news["date"] = pd.to_datetime(news["collect_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = df.merge(news, on="date", how="left")
    if "news_count" in merged.columns:
        merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    news_emb_cols = [c for c in merged.columns if c.startswith("news_emb_")]
    if news_emb_cols:
        merged[news_emb_cols] = merged[news_emb_cols].fillna(0)
    return merged


def _build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    anchor = pd.Timestamp("2000-01-01")
    global_times = sorted(df["time"].unique())
    time2idx = {t: i for i, t in enumerate(global_times)}
    df = df.copy()
    df["time_idx_int"] = df["time"].map(time2idx)
    df["time_idx"] = anchor + pd.to_timedelta(df["time_idx_int"].astype(int), unit="D")
    return df


def _ensure_log_return1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df.columns:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    calc = np.log(df["close"] / df["close"].shift(1))
    if "log_return_1" in df.columns:
        df["log_return_1"] = df["log_return_1"].fillna(calc)
    else:
        df["log_return_1"] = calc
    return df


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["rv_5"] = df["log_return_1"].rolling(5).std()
    df["rv_10"] = df["log_return_1"].rolling(10).std()

    for col in ["open", "high", "EMA", "EMA_5", "EMA_10"]:
        if col in df.columns:
            df[col] = df[col] / df["close"]

    exclude_cols = {
        "time",
        "item_id",
        "close",
        "time_idx",
        "time_idx_int",
        "EMA_20",
        "EMA_50",
        "EMA_100",
        "Volume",
        "symbol",
        "trade_date",
        "date",
        "collect_date",
        "embedding",
        "news_embedding_mean",
        "key_word",
        "vol_return_7d",
        "vol_return_14d",
        "vol_return_21d",
    }
    drop_keywords = ("created_at", "ingest_time")

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols or c.startswith("log_return_") or c.startswith("vol_return_"):
            continue
        if any(k in c for k in drop_keywords):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    numeric_cols = [c for c in (feature_cols + ["log_return_1"]) if c in df.columns]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df, feature_cols


def _load_scaler(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}. Retrain DeepAR to generate scaler.npz.")
    data = np.load(path, allow_pickle=True)
    return {
        "feature_names": data["feature_names"],
        "feature_mean": data["feature_mean"],
        "feature_std": data["feature_std"],
        "target_mean": data["target_mean"],
        "target_std": data["target_std"],
    }


def _apply_standard_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_col: str,
    target_mean: float,
    target_std: float,
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = (out[feature_cols].to_numpy(dtype=float) - feature_mean) / feature_std
    if target_col in out.columns:
        out[target_col] = (out[target_col].to_numpy(dtype=float) - target_mean) / target_std
    return out


def _build_infer_dataset(df: pd.DataFrame, target_col: str, feature_cols: List[str], prediction_length: int):
    df = df.sort_values("time_idx").reset_index(drop=True)
    target = df[target_col].to_numpy(dtype=float)
    feats = np.vstack([df[c].to_numpy(dtype=float) for c in feature_cols])
    if prediction_length > 0:
        pad = np.repeat(feats[:, -1:], prediction_length, axis=1)
        feats = np.concatenate([feats, pad], axis=1)
    start = df["time_idx"].iloc[0]
    return ListDataset(
        [{"target": target, "start": start, "feat_dynamic_real": feats}],
        freq="D",
    )


def _log_return_to_close(log_returns: np.ndarray, base_close: float) -> np.ndarray:
    close = np.zeros_like(log_returns, dtype=float)
    close[0] = base_close * np.exp(log_returns[0])
    for i in range(1, len(log_returns)):
        close[i] = close[i - 1] * np.exp(log_returns[i])
    return close


def _resolve_future_price_path(data_dir: str, commodity: str, future_price_file: str) -> Path | None:
    candidates = [
        Path(future_price_file.format(commodity=commodity)),
        Path(data_dir) / "future_price.csv",
        Path(data_dir).parents[1] / f"{commodity}_future_price.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _calendar_from_frames(
    infer_raw: pd.DataFrame | None,
    train_raw: pd.DataFrame | None,
    future_path: Path | None,
) -> List[str]:
    dates: List[str] = []
    for df in [infer_raw, train_raw]:
        if df is None:
            continue
        if "time" in df.columns:
            dates.extend(pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
    if future_path is not None and future_path.exists():
        fdf = pd.read_csv(future_path)
        if "time" in fdf.columns:
            dates.extend(pd.to_datetime(fdf["time"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
        elif "trade_date" in fdf.columns:
            dates.extend(pd.to_datetime(fdf["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
        elif "date" in fdf.columns:
            dates.extend(pd.to_datetime(fdf["date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist())
    dates = [d for d in dates if d and d != "NaT"]
    return sorted(set(dates))


def _resolve_exchange_for_commodity(commodity: Optional[str]) -> str:
    if not commodity:
        return "NYSE"
    key = str(commodity).strip()
    mapping = {
        "corn": "CBOT",
        "wheat": "CBOT",
        "soybean": "CBOT",
        "gold": "COMEX",
        "silver": "COMEX",
        "copper": "COMEX",
        "ZC=F": "CBOT",
        "ZW=F": "CBOT",
        "ZS=F": "CBOT",
        "GC=F": "COMEX",
        "SI=F": "COMEX",
        "HG=F": "COMEX",
    }
    if key in mapping:
        return mapping[key]
    lower = key.lower()
    if lower in mapping:
        return mapping[lower]
    return "NYSE"


def _exchange_future_dates(
    as_of: str,
    horizons: List[int],
    exchange: str = "NYSE",
) -> Dict[int, str]:
    if not horizons:
        return {}
    try:
        import pandas_market_calendars as mcal
    except Exception:
        return {}
    try:
        calendar = mcal.get_calendar(exchange)
    except Exception:
        return {}

    base = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(base):
        return {}
    max_h = max(int(h) for h in horizons)
    if max_h <= 0:
        return {}
    start = base + pd.Timedelta(days=1)
    end = base + pd.Timedelta(days=max_h * 5 + 7)
    try:
        valid_days = calendar.valid_days(start_date=start, end_date=end)
    except Exception:
        return {}
    day_list = [d.date().isoformat() for d in valid_days]
    mapping: Dict[int, str] = {}
    for h in horizons:
        idx = int(h) - 1
        if 0 <= idx < len(day_list):
            mapping[int(h)] = day_list[idx]
    return mapping


def _next_trading_dates(
    as_of: str,
    horizons: List[int],
    trading_calendar: List[str],
    exchange: str = "NYSE",
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if trading_calendar and as_of in trading_calendar:
        idx = trading_calendar.index(as_of)
        future = trading_calendar[idx + 1 :]
        for h in horizons:
            if h - 1 < len(future):
                mapping[h] = future[h - 1]
    missing = [int(h) for h in horizons if int(h) not in mapping]
    if missing:
        mapping.update(_exchange_future_dates(as_of, missing, exchange=exchange))
    base = pd.to_datetime(as_of)
    for h in horizons:
        if h not in mapping:
            mapping[h] = (base + BDay(h)).strftime("%Y-%m-%d")
    return mapping


def main(cfg: DeepARInferenceConfig) -> None:
    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = PREDICT_ROOT / path
        return str(path)

    cfg.data_dir = _resolve_path(cfg.data_dir)
    cfg.future_price_file = _resolve_path(cfg.future_price_file)
    cfg.checkpoint_root = _resolve_path(cfg.checkpoint_root)
    cfg.prediction_root = _resolve_path(cfg.prediction_root)
    cfg.combined_output = _resolve_path(cfg.combined_output)

    price_bq = _resolve_price_bq_settings(cfg)
    if price_bq.get("credentials_path") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = price_bq["credentials_path"]
    if price_bq.get("project_id"):
        cfg.bq_project_id = str(price_bq["project_id"])
    if price_bq.get("dataset_id"):
        cfg.bq_dataset_id = str(price_bq["dataset_id"])

    article_bq = _resolve_article_bq_settings(cfg)

    data_root = Path(cfg.data_dir).resolve()

    train_raw = _load_price_df(
        data_source=cfg.data_source,
        data_dir=cfg.data_dir,
        filename="train_price.csv",
        commodity=cfg.target_commodity,
        bq_project_id=cfg.bq_project_id,
        bq_dataset_id=cfg.bq_dataset_id,
        bq_table=cfg.bq_train_table,
    )
    infer_raw = None
    if cfg.data_source == "bigquery":
        try:
            infer_raw = _load_price_df(
                data_source=cfg.data_source,
                data_dir=cfg.data_dir,
                filename="inference_price.csv",
                commodity=cfg.target_commodity,
                bq_project_id=cfg.bq_project_id,
                bq_dataset_id=cfg.bq_dataset_id,
                bq_table=cfg.bq_inference_table,
            )
        except Exception:
            infer_raw = None
    else:
        infer_path = os.path.join(cfg.data_dir, "inference_price.csv")
        if os.path.exists(infer_path):
            infer_raw = _load_price_df(
                data_source=cfg.data_source,
                data_dir=cfg.data_dir,
                filename="inference_price.csv",
                commodity=cfg.target_commodity,
                bq_project_id=cfg.bq_project_id,
                bq_dataset_id=cfg.bq_dataset_id,
                bq_table=cfg.bq_inference_table,
            )

    news_df = None
    if article_bq.get("project_id") and article_bq.get("dataset_id"):
        news_df = load_news_features_bq(
            project_id=article_bq.get("project_id"),
            dataset_id=article_bq.get("dataset_id"),
            table=article_bq.get("daily_table") or "daily_summary",
            commodity=cfg.target_commodity,
            credentials_path=article_bq.get("credentials_path"),
        )

    if infer_raw is not None:
        df = pd.concat([train_raw, infer_raw], axis=0, ignore_index=True)
        if df["time"].duplicated().any():
            dup_dates = (
                df.loc[df["time"].duplicated(keep=False), "time"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            sample = ", ".join(dup_dates[:10])
            raise ValueError(
                "Duplicate dates across train/inference detected. "
                f"Sample duplicates: {sample}. "
                "Please ensure train_price and inference_price do not overlap."
            )
        df = df.sort_values("time")
    else:
        df = train_raw.copy()

    df = _merge_news(df, news_df)
    df = _ensure_log_return1(df)
    df, feature_cols = _prepare_features(df)
    df = _build_time_index(df)

    latest_infer_date = _fetch_latest_inference_date(
        data_source=cfg.data_source,
        data_root=data_root,
        project_id=cfg.bq_project_id,
        dataset_id=cfg.bq_dataset_id,
        table=cfg.bq_inference_table,
        credentials_path=price_bq.get("credentials_path"),
    )
    if latest_infer_date:
        as_of = latest_infer_date
    elif infer_raw is not None and not infer_raw.empty:
        as_of = str(pd.to_datetime(infer_raw["time"]).max())[:10]
    else:
        as_of = str(pd.to_datetime(train_raw["time"]).max())[:10]

    tag = f"_{cfg.output_tag}" if cfg.output_tag else ""
    ckpt_root = Path(cfg.checkpoint_root.format(commodity=cfg.target_commodity, date=as_of, tag=tag))
    pred_root = Path(cfg.prediction_root.format(commodity=cfg.target_commodity, date=as_of, tag=tag))
    combined_out = Path(cfg.combined_output.format(commodity=cfg.target_commodity, date=as_of, tag=tag))

    scaler = _load_scaler(ckpt_root / "scaler.npz")
    scaler_features = [str(x) for x in scaler["feature_names"].tolist()]
    missing = [c for c in scaler_features if c not in df.columns]
    if missing:
        raise ValueError(f"Scaler feature columns missing in data: {missing}")
    feature_cols = scaler_features
    feature_mean = scaler["feature_mean"]
    feature_std = scaler["feature_std"]
    target_mean = float(scaler["target_mean"][0])
    target_std = float(scaler["target_std"][0])

    df = _apply_standard_scaler(
        df,
        feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_col="log_return_1",
        target_mean=target_mean,
        target_std=target_std,
    )

    future_path = _resolve_future_price_path(cfg.data_dir, cfg.target_commodity, cfg.future_price_file)
    trading_calendar = _calendar_from_frames(infer_raw, train_raw, future_path)
    horizons = sorted({int(h) for h in cfg.horizons})
    date_map = _next_trading_dates(
        as_of,
        horizons,
        trading_calendar,
        exchange=_resolve_exchange_for_commodity(cfg.target_commodity),
    )

    combined_rows = []

    for ctx_len in cfg.seq_lengths:
        w_ckpt = ckpt_root / f"w{ctx_len}" / "best_model"
        if not w_ckpt.exists():
            raise FileNotFoundError(f"Predictor not found: {w_ckpt}")
        predictor = PyTorchPredictor.deserialize(w_ckpt)

        infer_source = df
        if infer_raw is not None and not infer_raw.empty:
            infer_start = pd.to_datetime(infer_raw["time"]).min()
            infer_block = df[df["time"] >= infer_start].copy()
            if len(infer_block) < ctx_len:
                pad_len = ctx_len - len(infer_block)
                train_tail = df[df["time"] < infer_start].tail(pad_len)
                infer_source = pd.concat([train_tail, infer_block], ignore_index=True)
            else:
                infer_source = infer_block

        infer_ds = _build_infer_dataset(infer_source, "log_return_1", feature_cols, cfg.prediction_length)
        forecast_it = predictor.predict(infer_ds, num_samples=cfg.num_samples)
        forecast = list(forecast_it)[0]

        base_close = float(infer_source.sort_values("time").iloc[-1]["close"])
        quantile_close = {}
        for q in cfg.quantiles:
            log_q = forecast.quantile(q)
            log_q = log_q * target_std + target_mean
            quantile_close[f"q{int(q*100):02d}"] = _log_return_to_close(log_q, base_close)

        q50 = quantile_close.get("q50")
        if q50 is None:
            qkeys = sorted(quantile_close.keys())
            q50 = quantile_close[qkeys[len(qkeys) // 2]]
        q10 = quantile_close.get("q10")
        if q10 is None:
            q10 = q50
        q90 = quantile_close.get("q90")
        if q90 is None:
            q90 = q50

        for h in horizons:
            idx = h - 1
            if idx >= len(q10) or idx >= len(q90):
                continue
            predict_date = date_map.get(h)
            if predict_date is None:
                continue
            combined_rows.append(
                {
                    "base_date": as_of,
                    "window_size": ctx_len,
                    "predict_date": predict_date,
                    "predicted_q10": float(q10[idx]),
                    "predicted_q90": float(q90[idx]),
                }
            )

        w_pred_root = pred_root / f"w{ctx_len}"
        w_pred_root.mkdir(parents=True, exist_ok=True)
        if q50 is not None and q10 is not None and q90 is not None:
            import matplotlib.pyplot as plt

            hist_df = infer_source.sort_values("time").tail(ctx_len)
            hist_dates = (
                pd.to_datetime(hist_df["time"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            hist_close = hist_df["close"].to_numpy(dtype=float)

            future_dates = [date_map[h] for h in horizons if h in date_map]
            pred_len = min(len(future_dates), len(q50), len(q10), len(q90))
            future_dates = future_dates[:pred_len]
            q50_plot = q50[:pred_len]
            q10_plot = q10[:pred_len]
            q90_plot = q90[:pred_len]

            x_hist = np.arange(len(hist_close))
            x_future = np.arange(len(hist_close), len(hist_close) + pred_len)

            plt.figure(figsize=(12, 4))
            if len(hist_close) > 0:
                plt.plot(x_hist, hist_close, color="black", marker="o", label="history")
            if pred_len > 0:
                plt.plot(x_future, q50_plot, color="tab:blue", marker="o", label="q50")
                plt.fill_between(x_future, q10_plot, q90_plot, color="tab:blue", alpha=0.2, label="q10-q90")

            base_x = len(hist_close) - 1
            if base_x >= 0:
                plt.axvline(base_x, color="tab:orange", linestyle="--", linewidth=1, label="base_date")

            plt.title(f"{cfg.target_commodity} Close Forecast (w{ctx_len})")
            plt.xlabel("date")
            plt.ylabel("close")

            all_dates = hist_dates + future_dates
            all_x = list(range(len(all_dates)))
            if all_dates:
                max_ticks = 10
                step = max(1, len(all_dates) // max_ticks)
                tick_idx = list(range(0, len(all_dates), step))
                if tick_idx[-1] != len(all_dates) - 1:
                    tick_idx.append(len(all_dates) - 1)
                tick_pos = [all_x[i] for i in tick_idx]
                tick_labels = [all_dates[i] for i in tick_idx]
                plt.xticks(tick_pos, tick_labels, rotation=45, ha="right")

            plt.legend()
            plt.tight_layout()
            plt.savefig(w_pred_root / f"{as_of}_forecast_q10_q90.png", dpi=150)
            plt.close()

    if combined_rows:
        combined_out.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(combined_rows)
        if combined_out.exists():
            df_prev = pd.read_csv(combined_out)
            dedupe_cols = ["base_date", "window_size", "predict_date"]
            df_prev = df_prev[~df_prev[dedupe_cols].apply(tuple, axis=1).isin(
                df_out[dedupe_cols].apply(tuple, axis=1)
            )]
            df_out = pd.concat([df_prev, df_out], ignore_index=True)
        df_out = df_out[
            ["base_date", "window_size", "predict_date", "predicted_q10", "predicted_q90"]
        ]
        df_out.to_csv(combined_out, index=False)
        print(f"✓ DeepAR predictions saved: {combined_out}")


if __name__ == "__main__":
    main(tyro.cli(DeepARInferenceConfig))
