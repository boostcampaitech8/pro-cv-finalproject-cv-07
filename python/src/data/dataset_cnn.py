import ast
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from src.data.bigquery_loader import COMMODITY_TO_SYMBOL, load_price_table

HORIZONS = list(range(1, 21))
# Volume-based auxiliary features are intentionally disabled.
VOLUME_COLUMNS: List[str] = []
IMAGE_MODE_TO_FOLDER = {
    "candle": "candlestick",
    "gaf": "GAF",
    "rp": "RP",
}


def split_records(
    records: Sequence[int],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split indices into train/valid/test lists (utility kept for compatibility).
    """
    records = list(records)
    total = len(records)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train_records = records[:train_end]
    valid_records = records[train_end:valid_end]
    test_records = records[valid_end:]

    return train_records, valid_records, test_records


def _load_fold_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Fold file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_embedding(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=np.float32)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        return np.asarray(parsed, dtype=np.float32)
    return np.asarray([value], dtype=np.float32)


def _load_news_features_df(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], int]:
    if df is None or df.empty:
        return {}, 512

    df = df.copy()
    if "date" not in df.columns:
        if "collect_date" in df.columns:
            df["date"] = pd.to_datetime(df["collect_date"], errors="coerce")
        else:
            raise ValueError("news features must include a 'date' or 'collect_date' column.")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    embeddings: List[np.ndarray] = []
    if "embedding" in df.columns or "embedding[512]" in df.columns:
        col = "embedding" if "embedding" in df.columns else "embedding[512]"
        for value in df[col].tolist():
            embeddings.append(_parse_embedding(value))
    else:
        emb_cols = [c for c in df.columns if c.startswith("news_emb_")]
        if emb_cols:
            embeddings = [df.loc[i, emb_cols].to_numpy(dtype=np.float32) for i in df.index]
        else:
            numeric_cols = [
                c for c in df.columns
                if c not in {"date", "collect_date", "news_count"}
                and np.issubdtype(df[c].dtype, np.number)
            ]
            embeddings = [df.loc[i, numeric_cols].to_numpy(dtype=np.float32) for i in df.index]

    max_dim = max((len(vec) for vec in embeddings), default=0)
    if max_dim == 0:
        return {}, 512

    padded_embeddings: List[np.ndarray] = []
    for vec in embeddings:
        if len(vec) < max_dim:
            pad = np.zeros(max_dim - len(vec), dtype=np.float32)
            vec = np.concatenate([vec, pad])
        padded_embeddings.append(vec.astype(np.float32))

    data = {date: vec for date, vec in zip(df["date"].tolist(), padded_embeddings)}
    return data, max_dim


def _load_news_features(path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    if not path.exists():
        return {}, 512

    df = pd.read_csv(path)
    return _load_news_features_df(df)


def _load_grayscale_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load images.") from exc

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        img = img.convert("L")
        array = np.asarray(img, dtype=np.float32) / 255.0

    return array


_GCS_CLIENT = None


def _get_gcs_client():
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError("google-cloud-storage is required to load images from GCS.") from exc
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def _load_grayscale_image_gcs(bucket_name: str, blob_name: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load images.") from exc

    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"Image not found in GCS: gs://{bucket_name}/{blob_name}")
    data = blob.download_as_bytes()
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("L")
        array = np.asarray(img, dtype=np.float32) / 255.0
    return array


class CNNDataset(Dataset):
    """
    CNN dataset for anomaly modeling with image, severity, and optional aux features.
    """

    _severity_cache: Dict[Tuple[str, int, int, str, str], Dict[str, np.ndarray]] = {}

    def __init__(
        self,
        commodity: str,
        fold: int,
        split: str,
        window_size: int,
        image_mode: str,
        use_aux: bool = False,
        aux_type: str = "news",
        data_dir: Optional[str] = None,
        split_file: Optional[str] = None,
        news_data: Optional[pd.DataFrame] = None,
        data_source: str = "local",
        bq_project_id: Optional[str] = None,
        bq_dataset_id: Optional[str] = None,
        bq_train_table: str = "train_price",
        bq_inference_table: str = "inference_price",
        bq_test_table: str = "test_price",
        image_source: str = "local",
        gcs_bucket: Optional[str] = None,
        gcs_prefix_template: str = "{symbol}/window_{window}_ohlc",
    ) -> None:
        if split not in {"train", "val", "test", "infer"}:
            raise ValueError("split must be 'train', 'val', 'test', or 'infer'.")
        if image_mode not in {"candle", "gaf", "rp", "stack", "candle_gaf", "candle_rp"}:
            raise ValueError("image_mode must be 'candle', 'gaf', 'rp', 'stack', 'candle_gaf', or 'candle_rp'.")
        if aux_type != "news":
            raise ValueError("aux_type must be 'news' (volume features disabled).")

        self.commodity = commodity
        self.fold = fold
        self.split = split
        self.window_size = window_size
        self.image_mode = image_mode
        self.use_aux = use_aux
        self.aux_type = aux_type
        self.data_source = data_source
        self.bq_project_id = bq_project_id
        self.bq_dataset_id = bq_dataset_id
        self.bq_train_table = bq_train_table
        self.bq_inference_table = bq_inference_table
        self.bq_test_table = bq_test_table
        self.image_source = image_source
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix_template = gcs_prefix_template
        self.symbol = COMMODITY_TO_SYMBOL.get(commodity.lower(), commodity)

        self.data_root = Path(data_dir) if data_dir else Path("src/datasets")
        self.preprocessing_root = self.data_root / "preprocessing"
        self.feature_path = self.preprocessing_root / f"{commodity}_feature_engineering.csv"
        if split_file:
            split_path = Path(split_file)
            if not split_path.exists() and not split_path.is_absolute():
                split_path = self.data_root / split_path
            self.fold_path = split_path
        else:
            if data_dir:
                self.fold_path = self.data_root / f"{commodity}_split.json"
            else:
                self.fold_path = self.data_root / "rolling_fold.json"
        if self.image_source == "gcs":
            self.image_root = None
            self.image_mode_to_folder = {}
        else:
            local_candle_root = self.data_root / "candle_img"
            if data_dir and local_candle_root.exists():
                self.image_root = self.data_root
                self.image_mode_to_folder = {
                    "candle": "candle_img",
                    "gaf": "gaf_img",
                    "rp": "rp_img",
                }
            else:
                self.image_root = self.preprocessing_root / f"{commodity}_cnn_preprocessing"
                self.image_mode_to_folder = IMAGE_MODE_TO_FOLDER
        self.news_path = self.data_root / "news_features.csv"
        self.news_data = news_data

        if self.data_source == "bigquery":
            self.df = self._load_from_bigquery()
        elif self.feature_path.exists():
            self.df = pd.read_csv(self.feature_path)
        else:
            self.df = self._load_from_price_splits()
        required_columns = {"time", "open", "high", "low", "close"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.df["time"] = pd.to_datetime(self.df["time"], errors="coerce")
        if self.df["time"].isna().any():
            raise ValueError("Failed to parse some values in 'time' column.")

        self.df = self.df.sort_values("time").reset_index(drop=True)
        self.df["date_str"] = self.df["time"].dt.strftime("%Y-%m-%d")

        self.dates = self.df["date_str"].tolist()
        self.date_to_index = {date: idx for idx, date in enumerate(self.dates)}

        self.log_returns, self.uses_precomputed_returns = self._load_log_returns(self.df)

        fold_data = _load_fold_json(self.fold_path)
        self.fold_dates = self._get_fold_dates(fold_data)
        if self.split == "infer":
            infer_dates = self._get_inference_dates(fold_data)
            self.anchor_indices = self._build_anchor_indices(
                infer_dates,
                require_future=False,
                require_returns=False,
            )
        elif self.split == "test":
            test_dates = self._get_test_dates(fold_data)
            self.anchor_indices = self._build_anchor_indices(
                test_dates,
                require_future=False,
                require_returns=True,
            )
        else:
            self.anchor_indices = self._build_anchor_indices(
                self.fold_dates[self.split],
                require_future=True,
                require_returns=True,
            )
        self.anchor_dates = [self.dates[idx] for idx in self.anchor_indices]

        self.anchor_returns = self._collect_anchor_returns(self.anchor_indices, fill_nan=True)

        self.q80, self.q90, self.q95 = self._get_severity_thresholds(fold_data)

        self.news_embeddings: Dict[str, np.ndarray] = {}
        self.news_dim = 0
        if self.use_aux and self.aux_type == "news":
            if self.news_data is not None:
                self.news_embeddings, self.news_dim = _load_news_features_df(self.news_data)
            else:
                self.news_embeddings, self.news_dim = _load_news_features(self.news_path)
            if self.news_dim == 0:
                self.news_dim = 512

    def _load_from_price_splits(self) -> pd.DataFrame:
        """
        Fallback loader: build a full price dataframe from train/test/inference splits.
        """
        candidates = [
            (self.data_root / "train_price.csv", "train"),
            (self.data_root / "test_price.csv", "test"),
            (self.data_root / "inference_price.csv", "inference"),
        ]
        frames: List[pd.DataFrame] = []
        for path, source in candidates:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "time" not in df.columns:
                if "date" in df.columns:
                    df = df.rename(columns={"date": "time"})
                elif "trade_date" in df.columns:
                    df = df.rename(columns={"trade_date": "time"})
            df["_source"] = source
            frames.append(df)

        if not frames:
            raise FileNotFoundError(
                f"Feature CSV not found: {self.feature_path} and no price splits in {self.data_root}"
            )

        merged = pd.concat(frames, axis=0, ignore_index=True)
        if "time" not in merged.columns:
            raise ValueError("Expected 'time' column in price splits.")
        merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
        merged = merged.dropna(subset=["time"]).sort_values("time")
        if merged["time"].duplicated().any():
            dup_dates = (
                merged.loc[merged["time"].duplicated(keep=False), "time"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            sample = ", ".join(dup_dates[:10])
            raise ValueError(
                "Duplicate dates across price splits detected. "
                f"Sample duplicates: {sample}. "
                "Please ensure train/test/inference splits are non-overlapping."
            )
        merged = merged.drop(columns=["_source"], errors="ignore").reset_index(drop=True)
        return merged

    def _load_from_bigquery(self) -> pd.DataFrame:
        """
        Load price data from BigQuery train/test/inference tables.
        """
        if not self.bq_project_id or not self.bq_dataset_id:
            raise ValueError("BigQuery project_id/dataset_id must be provided for data_source='bigquery'.")

        candidates = [
            (self.bq_train_table, "train"),
            (self.bq_test_table, "test"),
            (self.bq_inference_table, "inference"),
        ]
        frames: List[pd.DataFrame] = []
        for table, source in candidates:
            if not table:
                continue
            try:
                df = load_price_table(
                    project_id=self.bq_project_id,
                    dataset_id=self.bq_dataset_id,
                    table=table,
                    commodity=self.commodity,
                )
            except Exception as exc:
                print(f"⚠️  Skipping BigQuery table '{table}' (reason: {exc})")
                continue
            if df.empty:
                continue
            df["_source"] = source
            frames.append(df)

        if not frames:
            raise FileNotFoundError(
                "No price tables loaded from BigQuery. "
                f"Checked tables: {[c[0] for c in candidates if c[0]]}."
            )

        merged = pd.concat(frames, axis=0, ignore_index=True)
        if "time" not in merged.columns:
            raise ValueError("Expected 'time' column in BigQuery price tables.")
        merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
        merged = merged.dropna(subset=["time"]).sort_values("time")
        if merged["time"].duplicated().any():
            dup_dates = (
                merged.loc[merged["time"].duplicated(keep=False), "time"]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            sample = ", ".join(dup_dates[:10])
            raise ValueError(
                "Duplicate dates across BigQuery price tables detected. "
                f"Sample duplicates: {sample}. "
                "Please ensure train/test/inference tables are non-overlapping."
            )
        merged = merged.drop(columns=["_source"], errors="ignore").reset_index(drop=True)
        return merged

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, index: int) -> Dict:
        anchor_index = self.anchor_indices[index]
        anchor_date = self.anchor_dates[index]

        image = self._load_image(anchor_date)
        severity = self._build_severity(self.anchor_returns[index])
        aux = self._build_aux(anchor_index, anchor_date)

        return {
            "image": image,
            "aux": aux,
            "severity": severity,
            "meta": {
                "date": anchor_date,
                "fold": self.fold,
                "split": self.split,
                "window": self.window_size,
            },
        }

    def _compute_log_returns_from_close(self, close_prices: np.ndarray) -> Dict[int, np.ndarray]:
        log_returns: Dict[int, np.ndarray] = {}
        log_prices = np.log(close_prices + 1e-12)
        for h in HORIZONS:
            values = np.full_like(close_prices, np.nan, dtype=np.float32)
            if len(close_prices) > h:
                values[:-h] = np.abs(log_prices[h:] - log_prices[:-h])
            log_returns[h] = values
        return log_returns

    def _load_log_returns(self, df: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], bool]:
        """
        Prefer precomputed log_return_{h} columns if available; otherwise compute from close.
        Returns (log_returns, uses_precomputed).
        """
        close_prices = df["close"].to_numpy(dtype=np.float32)
        computed = self._compute_log_returns_from_close(close_prices)

        log_returns: Dict[int, np.ndarray] = {}
        uses_precomputed = False
        for h in HORIZONS:
            col = f"log_return_{h}"
            if col in df.columns:
                uses_precomputed = True
                log_returns[h] = df[col].to_numpy(dtype=np.float32)
            else:
                log_returns[h] = computed[h]
        return log_returns, uses_precomputed

    def _get_fold_dates(self, fold_data: Dict) -> Dict[str, List[str]]:
        folds = fold_data.get("folds", [])
        fold_entry = next((f for f in folds if f.get("fold") == self.fold), None)
        if fold_entry is None:
            raise ValueError(f"Fold {self.fold} not found in {self.fold_path}.")

        dates: Dict[str, List[str]] = {}
        for split in ("train", "val"):
            split_entry = fold_entry.get(split, {})
            t_dates = split_entry.get("t_dates")
            t_indices = split_entry.get("t_indices")

            if t_dates is not None:
                dates[split] = [str(d) for d in t_dates]
            elif t_indices is not None:
                dates[split] = [self.dates[idx] for idx in t_indices]
            else:
                raise ValueError(f"Missing t_dates/t_indices for fold {self.fold} {split}")

        return dates

    def _get_inference_dates(self, fold_data: Dict) -> List[str]:
        meta = fold_data.get("meta", {})
        inference_window = meta.get("inference_window", {})
        start = inference_window.get("start")
        end = inference_window.get("end")
        if not start or not end:
            raise ValueError("Missing inference_window in split meta for infer split.")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        dates = [
            d for d in self.dates if start_dt <= pd.to_datetime(d) <= end_dt
        ]
        if not dates:
            raise ValueError("No dates found in inference window.")
        return dates

    def _get_test_dates(self, fold_data: Dict) -> List[str]:
        meta = fold_data.get("meta", {})
        fixed_test = meta.get("fixed_test", {})
        t_dates = fixed_test.get("t_dates")
        t_indices = fixed_test.get("t_indices")
        if t_dates:
            return [str(d) for d in t_dates]
        if t_indices:
            return [self.dates[idx] for idx in t_indices]
        raise ValueError("Missing fixed_test t_dates/t_indices for test split.")

    def _build_anchor_indices(
        self,
        split_dates: Sequence[str],
        require_future: bool = True,
        require_returns: bool = True,
    ) -> List[int]:
        indices: List[int] = []
        max_horizon = max(HORIZONS)
        total = len(self.dates)

        for date_str in split_dates:
            if date_str not in self.date_to_index:
                continue
            idx = self.date_to_index[date_str]
            if idx < self.window_size - 1:
                continue
            if require_future and idx + max_horizon >= total:
                continue
            if require_returns:
                valid = True
                for h in HORIZONS:
                    value = self.log_returns[h][idx]
                    if not np.isfinite(value):
                        valid = False
                        break
                if not valid:
                    continue
            indices.append(idx)

        return indices

    def _collect_anchor_returns(self, indices: Sequence[int], fill_nan: bool = False) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros((0, len(HORIZONS)), dtype=np.float32)
        stacked = [self.log_returns[h][indices] for h in HORIZONS]
        out = np.stack(stacked, axis=1).astype(np.float32)
        if fill_nan:
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _get_severity_thresholds(
        self, fold_data: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (
            self.commodity,
            self.fold,
            self.window_size,
            str(self.data_root.resolve()),
            str(self.fold_path.resolve()),
        )
        if cache_key in self._severity_cache:
            cached = self._severity_cache[cache_key]
            return cached["q80"], cached["q90"], cached["q95"]

        train_dates = self._get_fold_dates(fold_data)["train"]
        train_indices = self._build_anchor_indices(train_dates)
        train_returns = self._collect_anchor_returns(train_indices)

        if train_returns.size == 0:
            raise ValueError("No training samples available to compute severity thresholds.")

        q80 = np.percentile(train_returns, 80, axis=0).astype(np.float32)
        q90 = np.percentile(train_returns, 90, axis=0).astype(np.float32)
        q95 = np.percentile(train_returns, 95, axis=0).astype(np.float32)

        self._severity_cache[cache_key] = {"q80": q80, "q90": q90, "q95": q95}
        return q80, q90, q95

    def _build_severity(self, returns: np.ndarray) -> torch.Tensor:
        """
        Build severity targets using raw log returns.

        Using raw abs log returns avoids distortion from extra normalization
        and keeps target scale consistent with the actual returns.
        """
        return torch.from_numpy(returns.astype(np.float32))

    def _load_image(self, date_str: str) -> torch.Tensor:
        if self.image_mode == "stack":
            candle = self._load_single_image("candle", date_str)
            gaf = self._load_single_image("gaf", date_str)
            rp = self._load_single_image("rp", date_str)
            image = np.stack([candle, gaf, rp], axis=0)
        elif self.image_mode == "candle_gaf":
            candle = self._load_single_image("candle", date_str)
            gaf = self._load_single_image("gaf", date_str)
            image = np.stack([candle, gaf], axis=0)
        elif self.image_mode == "candle_rp":
            candle = self._load_single_image("candle", date_str)
            rp = self._load_single_image("rp", date_str)
            image = np.stack([candle, rp], axis=0)
        else:
            single = self._load_single_image(self.image_mode, date_str)
            image = single[None, :, :]
        return torch.from_numpy(image.astype(np.float32))

    def _load_single_image(self, mode: str, date_str: str) -> np.ndarray:
        if self.image_source == "gcs":
            if mode != "candle":
                raise ValueError("GCS image source currently supports only candle images.")
            if not self.gcs_bucket:
                raise ValueError("gcs_bucket must be provided for image_source='gcs'.")
            prefix = self.gcs_prefix_template.format(
                symbol=self.symbol,
                commodity=self.commodity,
                window=self.window_size,
                mode=mode,
            ).strip("/")
            blob_name = f"{prefix}/{date_str}.png"
            return _load_grayscale_image_gcs(self.gcs_bucket, blob_name)

        folder = self.image_mode_to_folder.get(mode)
        if folder is None:
            raise ValueError(f"Unsupported image mode: {mode}")
        image_path = self.image_root / folder / f"w{self.window_size}" / f"{date_str}.png"
        return _load_grayscale_image(image_path)

    def _build_aux(self, anchor_index: int, anchor_date: str) -> Optional[torch.Tensor]:
        if not self.use_aux:
            return None

        aux_parts: List[np.ndarray] = []

        if self.aux_type == "news":
            embedding = self.news_embeddings.get(anchor_date)
            if embedding is None:
                embedding = np.zeros(self.news_dim, dtype=np.float32)
            aux_parts.append(embedding.astype(np.float32))

        if not aux_parts:
            return None

        aux_vector = np.concatenate(aux_parts).astype(np.float32)
        return torch.from_numpy(aux_vector)


def cnn_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function that safely handles None aux fields.
    """
    aux_list = [item.get("aux") for item in batch]
    cleaned: List[Dict] = []
    for item in batch:
        new_item = dict(item)
        new_item.pop("aux", None)
        cleaned.append(new_item)

    collated = default_collate(cleaned)

    if all(aux is None for aux in aux_list):
        collated["aux"] = None
        return collated

    ref = next(aux for aux in aux_list if aux is not None)
    if not torch.is_tensor(ref):
        ref = torch.as_tensor(ref)

    aux_tensors: List[torch.Tensor] = []
    for aux in aux_list:
        if aux is None:
            aux_tensors.append(torch.zeros_like(ref))
        else:
            aux_tensors.append(torch.as_tensor(aux))

    collated["aux"] = default_collate(aux_tensors)
    return collated


def _smoke_test() -> None:
    ds = CNNDataset(
        commodity="corn",
        fold=0,
        split="train",
        window_size=20,
        image_mode="candle",
        use_aux=False,
    )
    sample = ds[0]
    print(sample["image"].shape, sample["severity"])


if __name__ == "__main__":
    _smoke_test()
