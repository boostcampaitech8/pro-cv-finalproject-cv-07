import numpy as np
import json


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