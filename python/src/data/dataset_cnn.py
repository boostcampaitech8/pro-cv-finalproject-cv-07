import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


HORIZONS = [1, 5, 10, 20]
VOLUME_COLUMNS = [
    "Volume",
    "vol_return_7d",
    "vol_volume_7d",
    "vol_return_14d",
    "vol_volume_14d",
    "vol_return_21d",
    "vol_volume_21d",
]
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


def _load_news_features(path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    if not path.exists():
        return {}, 512

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("news_features.csv must include a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    embedding_cols = [c for c in df.columns if c != "date"]
    embeddings: List[np.ndarray] = []

    if "embedding" in df.columns or "embedding[512]" in df.columns:
        col = "embedding" if "embedding" in df.columns else "embedding[512]"
        for value in df[col].tolist():
            embeddings.append(_parse_embedding(value))
    else:
        embeddings = [df.loc[i, embedding_cols].to_numpy(dtype=np.float32) for i in df.index]

    max_dim = max((len(vec) for vec in embeddings), default=0)
    if max_dim == 0:
        return {}, 512

    padded_embeddings: List[np.ndarray] = []
    for vec in embeddings:
        if len(vec) < max_dim:
            pad = np.zeros(max_dim - len(vec), dtype=np.float32)
            vec = np.concatenate([vec, pad])
        padded_embeddings.append(vec.astype(np.float32))

    data = {
        date: vec
        for date, vec in zip(df["date"].tolist(), padded_embeddings)
    }
    return data, max_dim


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


class CNNDataset(Dataset):
    """
    CNN dataset for anomaly modeling with image, severity, and optional aux features.
    """

    _severity_cache: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}

    def __init__(
        self,
        commodity: str,
        fold: int,
        split: str,
        window_size: int,
        image_mode: str,
        use_aux: bool = False,
        aux_type: str = "volume",
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        if image_mode not in {"candle", "gaf", "rp", "stack"}:
            raise ValueError("image_mode must be 'candle', 'gaf', 'rp', or 'stack'.")
        if aux_type not in {"volume", "news", "both"}:
            raise ValueError("aux_type must be 'volume', 'news', or 'both'.")

        self.commodity = commodity
        self.fold = fold
        self.split = split
        self.window_size = window_size
        self.image_mode = image_mode
        self.use_aux = use_aux
        self.aux_type = aux_type

        self.data_root = Path("src/datasets")
        self.preprocessing_root = self.data_root / "preprocessing"
        self.feature_path = self.preprocessing_root / f"{commodity}_feature_engineering.csv"
        self.fold_path = self.data_root / "rolling_fold.json"
        self.image_root = self.preprocessing_root / f"{commodity}_cnn_preprocessing"
        self.news_path = self.data_root / "news_features.csv"

        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {self.feature_path}")

        self.df = pd.read_csv(self.feature_path)
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

        close_prices = self.df["close"].to_numpy(dtype=np.float32)
        self.log_returns = self._compute_log_returns(close_prices)

        fold_data = _load_fold_json(self.fold_path)
        self.fold_dates = self._get_fold_dates(fold_data)
        self.anchor_indices = self._build_anchor_indices(self.fold_dates[self.split])
        self.anchor_dates = [self.dates[idx] for idx in self.anchor_indices]

        self.anchor_returns = self._collect_anchor_returns(self.anchor_indices)

        self.q80, self.q90, self.q95 = self._get_severity_thresholds(fold_data)

        if self.use_aux and self.aux_type in {"volume", "both"}:
            missing_volume = [col for col in VOLUME_COLUMNS if col not in self.df.columns]
            if missing_volume:
                raise ValueError(
                    f"Missing volume columns for aux features: {missing_volume}"
                )

        self.news_embeddings: Dict[str, np.ndarray] = {}
        self.news_dim = 0
        if self.use_aux and self.aux_type in {"news", "both"}:
            self.news_embeddings, self.news_dim = _load_news_features(self.news_path)
            if self.news_dim == 0:
                self.news_dim = 512

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

    def _compute_log_returns(self, close_prices: np.ndarray) -> Dict[int, np.ndarray]:
        log_returns: Dict[int, np.ndarray] = {}
        log_prices = np.log(close_prices + 1e-12)
        for h in HORIZONS:
            values = np.full_like(close_prices, np.nan, dtype=np.float32)
            if len(close_prices) > h:
                values[:-h] = np.abs(log_prices[h:] - log_prices[:-h])
            log_returns[h] = values
        return log_returns

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

    def _build_anchor_indices(self, split_dates: Sequence[str]) -> List[int]:
        indices: List[int] = []
        max_horizon = max(HORIZONS)
        total = len(self.dates)

        for date_str in split_dates:
            if date_str not in self.date_to_index:
                continue
            idx = self.date_to_index[date_str]
            if idx < self.window_size - 1:
                continue
            if idx + max_horizon >= total:
                continue
            indices.append(idx)

        return indices

    def _collect_anchor_returns(self, indices: Sequence[int]) -> np.ndarray:
        if len(indices) == 0:
            return np.zeros((0, len(HORIZONS)), dtype=np.float32)
        stacked = [self.log_returns[h][indices] for h in HORIZONS]
        return np.stack(stacked, axis=1).astype(np.float32)

    def _get_severity_thresholds(
        self, fold_data: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (self.commodity, self.fold, self.window_size)
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
        denom = np.maximum(self.q95 - self.q80, 1e-8)
        severity = np.clip((returns - self.q80) / denom, 0.0, 1.0)
        return torch.from_numpy(severity.astype(np.float32))

    def _load_image(self, date_str: str) -> torch.Tensor:
        if self.image_mode == "stack":
            candle = self._load_single_image("candle", date_str)
            gaf = self._load_single_image("gaf", date_str)
            rp = self._load_single_image("rp", date_str)
            image = np.stack([candle, gaf, rp], axis=0)
        else:
            single = self._load_single_image(self.image_mode, date_str)
            image = single[None, :, :]
        return torch.from_numpy(image.astype(np.float32))

    def _load_single_image(self, mode: str, date_str: str) -> np.ndarray:
        folder = IMAGE_MODE_TO_FOLDER.get(mode)
        if folder is None:
            raise ValueError(f"Unsupported image mode: {mode}")
        image_path = self.image_root / folder / f"w{self.window_size}" / f"{date_str}.png"
        return _load_grayscale_image(image_path)

    def _build_aux(self, anchor_index: int, anchor_date: str) -> Optional[torch.Tensor]:
        if not self.use_aux:
            return None

        aux_parts: List[np.ndarray] = []

        if self.aux_type in {"volume", "both"}:
            window = self.df.iloc[anchor_index - self.window_size + 1 : anchor_index + 1]
            volume_values = window[VOLUME_COLUMNS].to_numpy(dtype=np.float32)
            # Mean-pool volume features over the lookback window.
            volume_features = volume_values.mean(axis=0)
            aux_parts.append(volume_features)

        if self.aux_type in {"news", "both"}:
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
