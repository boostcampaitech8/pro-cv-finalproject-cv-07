import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional


class TFTDataset(Dataset):
    """
    TFT(Temporal Fusion Transformer)를 위한 Dataset 클래스
    Multi-horizon 예측을 지원
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        seq_length: int,
        horizons: List[int] = [1, 5, 10, 20],
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'Close',
        scale: bool = True
    ):
        """
        Args:
            data: 시계열 데이터프레임
            seq_length: lookback window size
            horizons: 예측할 미래 시점들 (e.g., [1, 5, 10, 20])
            feature_columns: 사용할 feature 컬럼들 (None이면 자동 선택)
            target_column: target 가격 컬럼
            scale: 데이터 정규화 여부
        """
        self.seq_length = seq_length
        self.horizons = horizons
        self.target_column = target_column
        self.scale = scale
        
        # Feature columns 설정
        if feature_columns is None:
            # 'time'과 log_return_* 컬럼 제외
            self.feature_columns = [
                c for c in data.columns 
                if c not in ['time', 'date'] and not c.startswith('log_return_')
            ]
        else:
            self.feature_columns = feature_columns
        
        # 데이터 준비
        self.data = data.copy()
        self._prepare_data()
        
    def _prepare_data(self):
        """데이터 전처리 및 정규화"""
        # Time column 저장
        if 'time' in self.data.columns:
            self.times = self.data['time'].values
        else:
            self.times = self.data.index.values
        
        # Feature 데이터 추출
        self.features = self.data[self.feature_columns].values.astype(np.float32)
        
        # Target 데이터 추출 (multi-horizon)
        target_columns = [f'log_return_{h}' for h in self.horizons]
        
        # target 컬럼이 없으면 생성
        for h, col in zip(self.horizons, target_columns):
            if col not in self.data.columns:
                print(f"Warning: {col} not found in data. Creating from {self.target_column}")
                # log return 계산
                self.data[col] = np.log(
                    self.data[self.target_column].shift(-h) / self.data[self.target_column]
                )
        
        self.targets = self.data[target_columns].values.astype(np.float32)
        
        # 정규화 (optional)
        if self.scale:
            self.feature_mean = np.nanmean(self.features, axis=0, keepdims=True)
            self.feature_std = np.nanstd(self.features, axis=0, keepdims=True)
            self.feature_std[self.feature_std == 0] = 1.0  # avoid division by zero
            
            self.features = (self.features - self.feature_mean) / self.feature_std
        
        # 유효한 인덱스 계산 (NaN 제거)
        max_horizon = max(self.horizons)
        self.valid_indices = []
        
        for i in range(len(self.data) - self.seq_length - max_horizon):
            # Feature window 체크
            feature_window = self.features[i:i+self.seq_length]
            # Target 체크
            target_idx = i + self.seq_length
            target_values = self.targets[target_idx]
            
            # NaN이 없는 경우만 valid
            if not np.isnan(feature_window).any() and not np.isnan(target_values).any():
                self.valid_indices.append(i)
        
        print(f"Total samples: {len(self.data)}, Valid samples: {len(self.valid_indices)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - 'encoder_input': [seq_length, num_features]
                - 'targets': [num_horizons]
                - 'time': timestamp
        """
        real_idx = self.valid_indices[idx]
        
        # Encoder input: [seq_length, num_features]
        encoder_input = self.features[real_idx:real_idx + self.seq_length]
        
        # Target: [num_horizons]
        target_idx = real_idx + self.seq_length
        targets = self.targets[target_idx]
        
        # Time
        time = self.times[target_idx]
        
        return {
            'encoder_input': torch.FloatTensor(encoder_input),
            'targets': torch.FloatTensor(targets),
            'time': str(time) if isinstance(time, (pd.Timestamp, np.datetime64)) else time
        }
    
    def get_feature_names(self) -> List[str]:
        """Feature 이름 반환"""
        return self.feature_columns


def build_tft_dataset(
    time_series: pd.DataFrame,
    seq_length: int,
    horizons: List[int] = [1, 5, 10, 20],
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    TFT용 데이터셋 생성 - 뉴스 embedding 포함
    """
    
    # ===== 1. Feature 컬럼 선택 =====
    if feature_columns is None:
        feature_columns = [
            c for c in time_series.columns 
            if c not in ['time', 'date', 'news_embedding_mean'] and not c.startswith('log_return_')
        ]
    
    # 숫자형 컬럼만 선택
    numeric_columns = time_series[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = numeric_columns
    
    print(f"Basic features: {len(feature_columns)}")
    
    # ===== 2. 뉴스 embedding 처리 =====
    has_news_embedding = 'news_embedding_mean' in time_series.columns
    
    if has_news_embedding:
        # 첫 번째 non-null embedding으로 차원 확인
        sample_embedding = time_series['news_embedding_mean'].dropna().iloc[0]
        if isinstance(sample_embedding, np.ndarray):
            embedding_dim = len(sample_embedding)
            print(f"News embedding dimension: {embedding_dim}")
        else:
            has_news_embedding = False
            print("No valid news embeddings found")
    
    # ===== 3. 데이터 생성 =====
    dataX = []
    dataY = []
    dataT = []
    
    max_horizon = max(horizons)
    
    for i in range(len(time_series) - seq_length - max_horizon):
        # 기본 Features
        _x_basic = time_series.loc[i:i+seq_length-1, feature_columns].values
        
        # 뉴스 Embedding 추가
        if has_news_embedding:
            news_embeddings = []
            for j in range(i, i+seq_length):
                emb = time_series.loc[j, 'news_embedding_mean']
                if isinstance(emb, np.ndarray):
                    news_embeddings.append(emb)
                else:
                    # 없으면 zero vector
                    news_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            
            news_embeddings = np.array(news_embeddings)  # [seq_length, embedding_dim]
            
            # 기본 feature와 concat
            _x = np.concatenate([_x_basic, news_embeddings], axis=1)  # [seq_length, num_features + embedding_dim]
        else:
            _x = _x_basic
        
        # Targets
        target_idx = i + seq_length
        _y = time_series.loc[target_idx, [f'log_return_{h}' for h in horizons]].values
        
        # Time
        _t = time_series.loc[target_idx, 'time']
        
        # NaN 체크
        try:
            has_nan_x = np.isnan(_x.astype(float)).any()
            has_nan_y = np.isnan(_y.astype(float)).any()
            
            if not has_nan_x and not has_nan_y:
                dataX.append(_x.astype(np.float32))
                dataY.append(_y.astype(np.float32))
                dataT.append(_t)
        except (ValueError, TypeError):
            continue
    
    # Feature 이름 업데이트
    final_feature_names = feature_columns.copy()
    if has_news_embedding:
        final_feature_names.extend([f'news_emb_{i}' for i in range(embedding_dim)])
    
    print(f"Total features after adding embeddings: {len(final_feature_names)}")
    
    return (
        np.array(dataX, dtype=np.float32),
        np.array(dataY, dtype=np.float32),
        np.array(dataT),
        final_feature_names
    )


def train_valid_split_tft(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    split_file: str,
    fold_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    기존 train_valid_split 함수와 동일한 인터페이스
    
    Returns:
        trainX, trainY, validX, validY
    """
    with open(split_file, 'r') as file:
        data = json.load(file)
    
    train_dates = data['folds'][fold_index]['train']['t_dates']
    valid_dates = data['folds'][fold_index]['val']['t_dates']
    
    T = np.array([str(t)[:10] for t in T])
    train_dates = [str(d)[:10] for d in train_dates]
    valid_dates = [str(d)[:10] for d in valid_dates]
    
    train_mask = np.isin(T, train_dates)
    valid_mask = np.isin(T, valid_dates)
    
    trainX, trainY = X[train_mask], Y[train_mask]
    validX, validY = X[valid_mask], Y[valid_mask]
    
    return trainX, trainY, validX, validY


def test_split_tft(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    split_file: str
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    기존 test_split 함수와 동일한 인터페이스
    
    Returns:
        test_dates, testX, testY
    """
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
    여러 fold를 관리하는 DataLoader 래퍼
    """
    
    def __init__(
        self,
        data_path: str,
        split_file: str,
        seq_length: int,
        horizons: List[int],
        batch_size: int,
        num_workers: int = 4,
        feature_columns: Optional[List[str]] = None
    ):
        self.data_path = data_path
        self.split_file = split_file
        self.seq_length = seq_length
        self.horizons = horizons
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_columns = feature_columns
        
        # 데이터 로드
        self.data = pd.read_csv(data_path)
        
        # Dataset 빌드
        self.X, self.Y, self.T, self.feature_names = build_tft_dataset(
            self.data,
            seq_length,
            horizons,
            feature_columns
        )
        
        print(f"Loaded data from {data_path}")
        print(f"Total samples: {len(self.X)}")
        print(f"Feature dimension: {self.X.shape[-1]}")
        print(f"Number of horizons: {len(horizons)}")
    
    def get_fold_loaders(
        self,
        fold_index: int
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        특정 fold의 train/valid DataLoader 반환
        """
        trainX, trainY, validX, validY = train_valid_split_tft(
            self.X, self.Y, self.T, self.split_file, fold_index
        )
        
        # PyTorch Dataset으로 변환
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(trainX),
            torch.FloatTensor(trainY)
        )
        
        valid_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(validX),
            torch.FloatTensor(validY)
        )
        
        # DataLoader 생성
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, valid_loader
    
    def get_test_loader(self) -> Tuple[List[str], torch.utils.data.DataLoader]:
        """
        Test DataLoader 반환
        """
        test_dates, testX, testY = test_split_tft(
            self.X, self.Y, self.T, self.split_file
        )
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(testX),
            torch.FloatTensor(testY)
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return test_dates, test_loader


if __name__ == "__main__":
    # 테스트 코드
    import os
    
    data_path = "./src/datasets/corn_feature_engineering.csv"
    split_file = "./src/datasets/folds_2017_11_09.json"
    
    if os.path.exists(data_path) and os.path.exists(split_file):
        loader_manager = TFTDataLoader(
            data_path=data_path,
            split_file=split_file,
            seq_length=20,
            horizons=[1, 5, 10, 20],
            batch_size=64
        )
        
        # Fold 0의 데이터 로드
        train_loader, valid_loader = loader_manager.get_fold_loaders(fold_index=0)
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Valid batches: {len(valid_loader)}")
        
        # 첫 번째 배치 확인
        for batch_X, batch_Y in train_loader:
            print(f"\nBatch shape:")
            print(f"  X: {batch_X.shape}")  # [batch_size, seq_length, num_features]
            print(f"  Y: {batch_Y.shape}")  # [batch_size, num_horizons]
            break
    else:
        print("Data files not found. Please check paths.")
