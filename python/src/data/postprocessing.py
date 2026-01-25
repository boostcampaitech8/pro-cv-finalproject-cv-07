import pandas as pd
import numpy as np


def convert_close(df, test_dates, true, pred):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])

    test_start_date = pd.to_datetime(min(test_dates))

    prev_day_row = df[df['time'] < test_start_date].iloc[-1]
    last_close = prev_day_row['close']
    
    true_close = [last_close * np.exp(true[0])]
    for r in true[1:]:
        true_close.append(true_close[-1] * np.exp(r))
    true_close = np.array(true_close)

    pred_close = [last_close * np.exp(pred[0])]
    for r in pred[1:]:
        pred_close.append(pred_close[-1] * np.exp(r))
    pred_close = np.array(pred_close)
    
    return true_close, pred_close