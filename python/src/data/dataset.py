import numpy as np
import json


def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    dataT = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series.loc[i:i+seq_length-1, [c for c in time_series.columns if c != 'time']].values
        _y = time_series.loc[i+seq_length, 'log_return']
        _t = time_series.loc[i+seq_length, 'time']

        dataX.append(_x)
        dataY.append(_y)
        dataT.append(_t)

    return np.array(dataX), np.array(dataY), np.array(dataT)


