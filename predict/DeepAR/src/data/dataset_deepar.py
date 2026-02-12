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



def deepar_split(df, split_file, fold_idx, time_col="time"):
    with open(split_file, "r") as f:
        data = json.load(f)
    train_dates = pd.to_datetime(data['folds'][fold_idx]['train']['t_dates'])
    valid_dates = pd.to_datetime(data['folds'][fold_idx]['val']['t_dates'])

    
    val_dates = pd.to_datetime(data["folds"][fold_idx]["val"]["t_dates"])
    val_start = val_dates.min()
    train_df = df[df["time"].isin(train_dates)].reset_index(drop=True)
    val_df   = df[df["time"].isin(valid_dates)].reset_index(drop=True)
    return train_df, val_df


def lag_features_by_1day(df: pd.DataFrame, feature_cols, group_col="item_id", time_col="time"):
    df = df.sort_values([group_col, time_col]).copy()

    # df에 실제 존재하는 컬럼만 선택
    existing_cols = [c for c in feature_cols if c in df.columns]

    # feature_cols만 1일 lag (누수 방지)
    df[existing_cols] = df.groupby(group_col)[existing_cols].shift(1)
    is_first = df.groupby(group_col).cumcount() == 0
    df = df.loc[~is_first].reset_index(drop=True)

    # lag로 생긴 NaN 제거(첫 날)
    #df = df.dropna(subset=existing_cols).reset_index(drop=True)
    return df



def build_multi_item_dataset(dfs, target_col, feature_cols):
    all_dfs = []
    min_len = min(len(df) for df in dfs.values())
    for item_id, df in dfs.items():
        df = df.copy()

        # 필수 컬럼 체크
        assert "item_id" in df.columns, f"{item_id}: item_id missing"
        assert target_col in df.columns, f"{item_id}: target missing"

        print(f"Item: {item_id}, rows={len(df)}, n_feat={len(feature_cols)}")
        #existing_features = [c for c in feature_cols if c in df.columns]

        
       

        all_dfs.append(df)
      

    # multi-item concat
    long_df = pd.concat(all_dfs, ignore_index=True)
    
    long_df = long_df.sort_values(["item_id", "time_idx"]).reset_index(drop=True)

    # PandasDataset 생성
    dataset = PandasDataset.from_long_dataframe(
        long_df,
        item_id="item_id",
        timestamp="time_idx",
        target=target_col,
        freq="D",  
        feat_dynamic_real=feature_cols,
    )

    return dataset
