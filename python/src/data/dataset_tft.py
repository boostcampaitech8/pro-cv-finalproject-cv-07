import numpy as np
import pandas as pd
import json
import random
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from bisect import bisect_right
from pandas.api.types import is_numeric_dtype


def _seed_worker(worker_id: int) -> None:
    # Ensure each worker has a deterministic but unique seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SimpleYScaler:
    """y_mean, y_std를 sklearn scaler처럼 사용"""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean  # [H]
        self.std = std    # [H]
    
    def transform(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.mean) / self.std).astype(np.float32)
    
    def inverse_transform(self, Y_scaled: np.ndarray) -> np.ndarray:
        return (Y_scaled * self.std + self.mean).astype(np.float32)

def _fit_x_scaler(trainX: np.ndarray, feature_mask: np.ndarray, eps: float = 1e-8):
    """
    trainX: [N, T, F]
    feature_mask: [F] boolean (True인 feature만 mean/std 학습해서 스케일링)
    """
    N, T, F = trainX.shape
    flat = trainX.reshape(-1, F)  # [N*T, F]

    # 마스크된 feature만 통계 계산
    flat_m = flat[:, feature_mask]
    mean_m = flat_m.mean(axis=0)
    std_m = flat_m.std(axis=0)
    std_m = np.where(std_m < eps, 1.0, std_m)

    # 전체 길이(F)로 mean/std 만들고,
    # 스케일 안 하는 feature는 (mean=0, std=1)로 둬서 transform 시 원본 유지
    mean = np.zeros(F, dtype=np.float32)
    std = np.ones(F, dtype=np.float32)
    mean[feature_mask] = mean_m.astype(np.float32)
    std[feature_mask] = std_m.astype(np.float32)

    return mean.astype(np.float32), std.astype(np.float32)


def _transform_x(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)

def _fit_y_scaler(trainY: np.ndarray, eps: float = 1e-8):
    """
    trainY: [N, H]  (H = num_horizons)
    """
    mean = trainY.mean(axis=0)
    std = trainY.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)

def _transform_y(Y: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((Y - mean) / std).astype(np.float32)

def _inverse_y(Y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (Y_scaled * std + mean).astype(np.float32)


def align_news_to_trading_dates_runtime(
    price_df: pd.DataFrame,
    news_df: pd.DataFrame,
    price_date_column: str = 'time'
) -> pd.DataFrame:
    """
    런타임에 뉴스를 거래일에 맞춰 정렬
    휴장일 뉴스는 이전 거래일과 mean pooling
    
    Args:
        price_df: 가격 데이터 (거래일 정보)
        news_df: 뉴스 데이터 (모든 날짜)
        price_date_column: 가격 데이터의 날짜 컬럼명
        
    Returns:
        거래일에 맞춰 정렬된 뉴스 DataFrame
    """
    # 거래일 추출
    if price_date_column in price_df.columns:
        trading_dates = sorted(pd.to_datetime(price_df[price_date_column]).dt.date.unique())
    elif 'date' in price_df.columns:
        trading_dates = sorted(pd.to_datetime(price_df['date']).dt.date.unique())
    else:
        raise ValueError("가격 데이터에 날짜 컬럼(time 또는 date)이 없습니다")
    
    trading_dates_set = set(trading_dates)
    
    # 뉴스를 날짜별 딕셔너리로 변환 (휴장일/주말은 직전 거래일로 보정)
    news_df_copy = news_df.copy()
    date_col = 'collect_date' if 'collect_date' in news_df_copy.columns else 'date'
    if date_col not in news_df_copy.columns:
        raise ValueError("뉴스 데이터에 날짜 컬럼(collect_date 또는 date)이 없습니다")

    # BQ 뉴스 스키마: news_embedding_mean (REPEATED) -> news_emb_* 컬럼으로 확장
    if "news_embedding_mean" in news_df_copy.columns and not any(
        c.startswith("news_emb_") for c in news_df_copy.columns
    ):
        emb_series = news_df_copy["news_embedding_mean"].dropna()
        if len(emb_series) > 0:
            first = emb_series.iloc[0]
            try:
                emb_len = len(first)
            except TypeError:
                emb_len = 512
        else:
            emb_len = 512

        emb_cols = [f"news_emb_{i}" for i in range(emb_len)]

        if len(news_df_copy) > 0:
            def _normalize_emb(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    if len(x) == emb_len:
                        return list(x)
                    if len(x) > emb_len:
                        return list(x)[:emb_len]
                    return list(x) + [0.0] * (emb_len - len(x))
                return [0.0] * emb_len

            emb_df = pd.DataFrame(
                news_df_copy["news_embedding_mean"].apply(_normalize_emb).tolist(),
                columns=emb_cols
            )
        else:
            emb_df = pd.DataFrame(columns=emb_cols)

        news_df_copy = news_df_copy.drop(columns=["news_embedding_mean"]).reset_index(drop=True)
        news_df_copy = pd.concat([news_df_copy, emb_df], axis=1)

    news_df_copy['date'] = pd.to_datetime(news_df_copy[date_col]).dt.date

    def _prev_trading_day(d: pd.Timestamp | None) -> pd.Timestamp | None:
        if d is None or pd.isna(d):
            return None
        if d in trading_dates_set:
            return d
        idx = bisect_right(trading_dates, d) - 1
        if idx < 0:
            return None
        return trading_dates[idx]

    news_df_copy['date'] = news_df_copy['date'].apply(_prev_trading_day)
    news_df_copy = news_df_copy[news_df_copy['date'].notna()]
    
    news_emb_cols = [col for col in news_df_copy.columns if col.startswith('news_emb_')]
    stat_cols = [
        c for c in news_df_copy.columns
        if c not in {date_col, "date", "news_count"}
        and not c.startswith("news_emb_")
    ]
    # Keep only numeric stats
    stat_cols = [c for c in stat_cols if is_numeric_dtype(news_df_copy[c])]
    for col in stat_cols:
        news_df_copy[col] = pd.to_numeric(news_df_copy[col], errors="coerce")

    if "news_count" not in news_df_copy.columns:
        news_df_copy["news_count"] = 0

    agg = {"news_count": "sum"}
    for col in news_emb_cols:
        agg[col] = "mean"
    for col in stat_cols:
        agg[col] = "mean"

    grouped = news_df_copy.groupby("date", as_index=False).agg(agg)
    grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce")
    grouped = grouped.dropna(subset=["date"])

    final_news_df = grouped.copy()
    if final_news_df.empty:
        base_cols = ["date", "news_count"]
        final_news_df = pd.DataFrame(columns=base_cols + stat_cols + news_emb_cols)
    
    # 모든 거래일에 대해 뉴스 데이터 생성
    all_trading_dates_df = pd.DataFrame({'date': [pd.Timestamp(d) for d in trading_dates]})
    
    result_df = all_trading_dates_df.merge(
        final_news_df,
        on='date',
        how='left'
    )
    
    # 뉴스가 없는 날 처리
    result_df['news_count'] = result_df['news_count'].fillna(0).astype(int)
    
    # news_emb/통계 컬럼들을 0으로 채우기
    for col in news_emb_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(0)
    for col in stat_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(0)
    
    return result_df


def load_data_with_news(
    price_data_path: "str | pd.DataFrame",
    news_data_path: "str | pd.DataFrame"
) -> pd.DataFrame:
    """
    가격 데이터와 뉴스 데이터를 로드해서 병합
    뉴스는 런타임에 거래일에 맞춰 자동 정렬
    
    Args:
        price_data_path: 가격 데이터 CSV 경로
        news_data_path: 뉴스 데이터 CSV 경로 (통합 파일)
        
    Returns:
        병합된 DataFrame
    """
    # 가격 데이터 로드
    if isinstance(price_data_path, pd.DataFrame):
        price_df = price_data_path.copy()
    else:
        price_df = pd.read_csv(price_data_path)
    print(f"✓ Price data loaded: {len(price_df)} rows, {len(price_df.columns)} columns")

    # 중복 날짜 체크 (엄격하게 에러 처리)
    date_col = None
    if "time" in price_df.columns:
        date_col = "time"
    elif "date" in price_df.columns:
        date_col = "date"
    if date_col is None:
        raise ValueError("가격 데이터에 날짜 컬럼(time 또는 date)이 없습니다")
    date_key = pd.to_datetime(price_df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    dup_mask = date_key.duplicated()
    if dup_mask.any():
        dup_dates = date_key[dup_mask].dropna().unique().tolist()
        sample = ", ".join(dup_dates[:5])
        raise ValueError(f"중복 날짜가 존재합니다: {sample}")
    
    # 뉴스 데이터 로드
    try:
        if isinstance(news_data_path, pd.DataFrame):
            news_df = news_data_path.copy()
        else:
            news_df = pd.read_csv(news_data_path)
        print(f"✓ News data loaded: {len(news_df)} rows, {len(news_df.columns)} columns")
        
        # 거래일에 맞춰 뉴스 정렬
        print(f"  → Aligning news to trading dates...")
        aligned_news = align_news_to_trading_dates_runtime(price_df, news_df)
        print(f"  ✓ Aligned: {len(aligned_news)} trading dates")
        
        # Date 컬럼 통일
        if 'time' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['time']).dt.date
            price_df['date'] = pd.to_datetime(price_df['date'])
        
        aligned_news['date'] = pd.to_datetime(aligned_news['date'])
        
        # Merge
        merged_df = price_df.merge(aligned_news, on='date', how='left')
        print(f"✓ Merged data: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        
        # 뉴스 없는 날 처리
        if 'news_count' in merged_df.columns:
            merged_df['news_count'] = merged_df['news_count'].fillna(0).astype(int)
        
        news_emb_cols = [col for col in merged_df.columns if col.startswith('news_emb_')]
        if news_emb_cols:
            merged_df[news_emb_cols] = merged_df[news_emb_cols].fillna(0)
            print(f"✓ Found {len(news_emb_cols)} news embedding columns")
        
        return merged_df
        
    except FileNotFoundError:
        print(f"⚠️  News data not found at {news_data_path}")
        print(f"   Proceeding without news features...")
        return price_df


def build_tft_dataset(
    time_series: pd.DataFrame,
    seq_length: int,
    horizons: List[int],
    feature_columns: Optional[List[str]] = None,
    require_targets: bool = True
):
    # Ensure numeric dtypes for feature/target columns
    time_series = time_series.copy()
    for col in time_series.columns:
        if col in ("time", "date"):
            continue
        time_series[col] = pd.to_numeric(time_series[col], errors="coerce")

    if feature_columns is None:
        exclude_cols = ['time', 'date']
        exclude_cols += [c for c in time_series.columns if c.startswith('log_return_')]
        numeric_cols = [
            c for c in time_series.columns
            if c not in exclude_cols and
            np.issubdtype(time_series[c].dtype, np.number)
        ]
        if require_targets:
            numeric_cols = [c for c in numeric_cols if time_series[c].notna().any()]

        news_emb_cols = sorted([c for c in numeric_cols if c.startswith('news_emb_')])
        other_numeric_cols = [c for c in numeric_cols if not c.startswith('news_emb_')]

        feature_columns = other_numeric_cols + news_emb_cols

    if not feature_columns:
        raise ValueError("No numeric features found after preprocessing. Check input CSV dtypes.")


    dataX, dataY, dataT = [], [], []
    max_h = max(horizons) if require_targets else 0
    if require_targets:
        end_idx = len(time_series) - seq_length - max_h
    else:
        end_idx = len(time_series) - seq_length

    for i in range(max(0, end_idx)):
        # 입력: t-seq_length ~ t-1 (target_base = i+seq_length)
        x = time_series.iloc[i:i+seq_length][feature_columns].values

        if not np.isfinite(x).all():
            if require_targets:
                continue
            # Inference: keep feature shape consistent, replace NaNs with 0
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        target_base = i + seq_length

        y = []
        valid = True

        if require_targets:
            for h in horizons:
                col = f'log_return_{h}'
                if col not in time_series.columns:
                    valid = False
                    break
                val = time_series.loc[target_base, col]

                if (
                    pd.isna(val)
                    or np.isinf(val)
                    or abs(val) > 1.0      # log_return ±100% 컷
                ):
                    valid = False
                    break

                y.append(val)

            if not valid:
                continue
        else:
            y = [0.0 for _ in horizons]

        t = time_series.loc[target_base, 'time']

        dataX.append(x.astype(np.float32))
        dataY.append(np.array(y, dtype=np.float32))
        dataT.append(t)

    return (
        np.array(dataX),
        np.array(dataY),
        np.array(dataT),
        feature_columns
    )


def train_valid_split_tft(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    split_file: str,
    fold_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/Valid split"""
    with open(split_file, 'r') as file:
        data = json.load(file)
    
    train_dates = data['folds'][fold_index]['train']['t_dates']
    valid_dates = data['folds'][fold_index]['val']['t_dates']
    
    T = np.array([str(t)[:10] for t in T])
    train_dates = [str(d)[:10] for d in train_dates]
    valid_dates = [str(d)[:10] for d in valid_dates]
    
    train_mask = np.isin(T, train_dates)
    valid_mask = np.isin(T, valid_dates)
    
    trainX, trainY, trainT = X[train_mask], Y[train_mask], T[train_mask]
    validX, validY, validT = X[valid_mask], Y[valid_mask], T[valid_mask]
    
    return trainX, trainY, trainT, validX, validY, validT


def test_split_tft(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    split_file: str
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Test split"""
    with open(split_file, 'r') as file:
        data = json.load(file)
    
    test_dates = data['meta']['fixed_test']['t_dates']
    
    T = np.array([str(t)[:10] for t in T])
    test_dates_formatted = [str(d)[:10] for d in test_dates]
    
    test_mask = np.isin(T, test_dates_formatted)
    
    testX, testY = X[test_mask], Y[test_mask]
    
    return test_dates, testX, testY


class TFTDataLoader:
    """
    DataLoader - 공통 뉴스 파일 사용
    """
    
    def __init__(
        self,
        price_data_path: str,
        news_data_path: str,
        split_file: str,
        seq_length: int,
        horizons: List[int],
        batch_size: int,
        num_workers: int = 4,
        feature_columns: Optional[List[str]] = None,
        require_targets: bool = True,
        seed: Optional[int] = None
    ):
        self.price_data_path = price_data_path
        self.news_data_path = news_data_path
        self.split_file = split_file
        self.seq_length = seq_length
        self.horizons = horizons
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_columns = feature_columns
        self.require_targets = require_targets
        self.seed = seed
        
        # 데이터 로드 (런타임에 정렬)
        self.data = load_data_with_news(price_data_path, news_data_path)
        
        # Dataset 빌드
        self.X, self.Y, self.T, self.feature_names = build_tft_dataset(
            self.data,
            seq_length,
            horizons,
            feature_columns,
            require_targets=require_targets
        )

        self.fold_scalers = {}
        
        print(f"\n✓ Total samples: {len(self.X)}")
        print(f"✓ Feature dimension: {self.X.shape[-1]}")
        print(f"✓ Number of horizons: {len(horizons)}")
    
    def get_fold_loaders(
        self,
        fold_index: int,
        scale_x: bool = True,
        scale_y: bool = True,
        eps: float = 1e-8
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """특정 fold의 train/valid DataLoader"""
        trainX, trainY, trainT, validX, validY, validT = train_valid_split_tft(
            self.X, self.Y, self.T, self.split_file, fold_index
        )
        scaler = {}
        # Save feature names so inference can match training feature order/count.
        scaler["feature_names"] = np.array(self.feature_names, dtype=object)

        if scale_x:
            # news_emb_* 는 스케일링 제외
            feature_mask = np.array(
                [not str(n).startswith("news_emb_") for n in self.feature_names],
                dtype=bool
            )

            x_mean, x_std = _fit_x_scaler(trainX, feature_mask=feature_mask, eps=eps)
            trainX = _transform_x(trainX, x_mean, x_std)
            validX = _transform_x(validX, x_mean, x_std)

            scaler["x_mean"] = x_mean
            scaler["x_std"] = x_std

            # (선택) 재현/디버깅용 메타 저장
            scaler["x_feature_mask"] = feature_mask.astype(np.int8)
            scaler["x_scaled_features"] = np.array(
                [self.feature_names[i] for i in np.where(feature_mask)[0]],
                dtype=object
            )

        if scale_y:
            y_mean, y_std = _fit_y_scaler(trainY, eps=eps)
            trainY = _transform_y(trainY, y_mean, y_std)
            validY = _transform_y(validY, y_mean, y_std)
            
            # SimpleYScaler 객체로 저장
            scaler["y_scaler_obj"] = SimpleYScaler(y_mean, y_std)
            scaler["y_mean"] = y_mean  # 호환성 유지
            scaler["y_std"] = y_std

        scaler["scale_x"] = bool(scale_x)
        scaler["scale_y"] = bool(scale_y)

        self.fold_scalers[fold_index] = scaler

        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(trainX),
            torch.FloatTensor(trainY)
        )
        
        valid_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(validX),
            torch.FloatTensor(validY)
        )
        
        generator = None
        worker_init_fn = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(fold_index))
            worker_init_fn = _seed_worker

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        
        return train_loader, valid_loader, validT
    
    def get_test_loader(
        self,
        fold_index: int,
        scale_x: bool = True,
        scale_y: bool = True
    ) -> Tuple[List[str], torch.utils.data.DataLoader]:
        """Test DataLoader"""
        test_dates, testX, testY = test_split_tft(
            self.X, self.Y, self.T, self.split_file
        )

        # fold scaler 적용 (train에서 fit된 값)
        scaler = self.fold_scalers.get(fold_index, None)

        if scaler is not None and scale_x and scaler.get("scale_x", False):
            testX = _transform_x(testX, scaler["x_mean"], scaler["x_std"])

        if scaler is not None and scale_y and scaler.get("scale_y", False):
            testY = _transform_y(testY, scaler["y_mean"], scaler["y_std"])

        
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(testX),
            torch.FloatTensor(testY)
        )
        
        generator = None
        worker_init_fn = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(fold_index) + 1000)
            worker_init_fn = _seed_worker

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        
        return test_dates, test_loader

    def get_inference_loader(
        self,
        scale_x: bool = True,
        scale_y: bool = True
    ) -> Tuple[List[str], torch.utils.data.DataLoader]:
        """Inference DataLoader (no split)"""
        X, Y = self.X, self.Y
        if scale_x and self.fold_scalers:
            scaler = next(iter(self.fold_scalers.values()))
            if scaler.get("scale_x", False):
                X = _transform_x(X, scaler["x_mean"], scaler["x_std"])
        if scale_y and self.fold_scalers:
            scaler = next(iter(self.fold_scalers.values()))
            if scaler.get("scale_y", False):
                Y = _transform_y(Y, scaler["y_mean"], scaler["y_std"])

        dates = [str(t)[:10] for t in self.T]
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(Y)
        )
        generator = None
        worker_init_fn = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + 2000)
            worker_init_fn = _seed_worker
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        return dates, loader
