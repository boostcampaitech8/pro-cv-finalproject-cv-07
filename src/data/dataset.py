import numpy as np
import json
from gluonts.dataset.pandas import PandasDataset
import pandas as pd

def build_dataset(time_series, seq_length, horizons=[1, 5, 10, 20]):
    dataX = []
    dataY = []
    dataT = []
    for i in range(0, len(time_series) - seq_length):
        
        _x = time_series.loc[i:i+seq_length-1, [c for c in time_series.columns if c != 'time']].values
        _y = time_series.loc[i+seq_length, [f'log_return_{h}' for h in horizons]]
        _t = time_series.loc[i+seq_length, 'time']

        dataX.append(_x)
        dataY.append(_y)
        dataT.append(_t)

    return np.array(dataX), np.array(dataY), np.array(dataT)


def train_valid_split(X, Y, T, split_file, index):
    with open(split_file, 'r') as file:
        data = json.load(file)
    
    train_dates = data['folds'][index]['train']['t_dates']
    valid_dates =  data['folds'][index]['val']['t_dates']
    
    T = np.array([str(t)[:10] for t in T])

    train_dates = [str(d)[:10] for d in train_dates]
    valid_dates = [str(d)[:10] for d in valid_dates]

    trainX, trainY = X[np.isin(T, train_dates)], Y[np.isin(T, train_dates)]
    validX, validY = X[np.isin(T, valid_dates)], Y[np.isin(T, valid_dates)]
    
    return trainX, trainY, validX, validY


def test_split(X, Y, T, split_file):
    with open(split_file, 'r') as file:
        data = json.load(file)

    test_dates = data['meta']['fixed_test']['t_dates']
    
    T = np.array([str(t)[:10] for t in T])
    test_dates  = [str(d)[:10] for d in test_dates]
    
    testX, testY = X[np.isin(T, test_dates)], Y[np.isin(T, test_dates)]
    
    return test_dates, testX, testY



def deepar_split(df, split_file, fold_idx):
    with open(split_file, "r") as f:
        data = json.load(f)

    train_dates = set(str(d)[:10] for d in data["folds"][fold_idx]["train"]["t_dates"])
    val_dates   = set(str(d)[:10] for d in data["folds"][fold_idx]["val"]["t_dates"])

    df = df.copy()
    df["date_str"] = df["time"].astype(str).str[:10]

    train_df = df[df["date_str"].isin(train_dates)].drop(columns="date_str")
    val_df   = df[df["date_str"].isin(val_dates)].drop(columns="date_str")
    
    train_df = (
        train_df
        .sort_values(["item_id", "time"])
        .reset_index(drop=True)
    )
    train_df["time"] = train_df.groupby("item_id").cumcount()

    val_df = (
        val_df
        .sort_values(["item_id", "time"])
        .reset_index(drop=True)
    )
    val_df["time"] = val_df.groupby("item_id").cumcount()

    return train_df, val_df




def build_multi_item_dataset(dfs, target_col, feature_cols):
    all_dfs = []
    min_len = min(len(df) for df in dfs.values())
    for item_id, df in dfs.items():
        df = df.copy()
        #df = df.iloc[-min_len:]

        # 필수 컬럼 체크
        assert "item_id" in df.columns, f"{item_id}: item_id missing"
        assert target_col in df.columns, f"{item_id}: target missing"

    
        # 필요한 컬럼만 선택
        keep_cols = ["time", "item_id", target_col] + feature_cols
        df = df[keep_cols]

        all_dfs.append(df)
        print(f"Item: {item_id}, Target: {len(df.values)}")

    # multi-item concat
    long_df = pd.concat(all_dfs, ignore_index=True)

    # PandasDataset 생성
    dataset = PandasDataset.from_long_dataframe(
        long_df,
        item_id="item_id",
        timestamp="time",
        target=target_col,
        freq=None,   # ← 숫자 index일 때는 None이 안전
        feat_dynamic_real=feature_cols,
    )
    

    return dataset