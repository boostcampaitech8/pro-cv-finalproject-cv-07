import pandas as pd
import numpy as np


def convert_close(df, test_dates, true, pred, horizons):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])

    test_start_date = pd.to_datetime(min(test_dates))

    prev_day_row = df[df['time'] < test_start_date].iloc[-1]
    last_close = prev_day_row['close']
    
    true_close_h = []
    pred_close_h = []
    
    for h in range(len(horizons)):
        t_seq = [last_close * np.exp(true[0, h])]
        for r in true[1:, h]:
            t_seq.append(t_seq[-1] * np.exp(r))
        true_close_h.append(np.array(t_seq))
        
        p_seq = [last_close * np.exp(pred[0, h])]
        for r in pred[1:, h]:
            p_seq.append(p_seq[-1] * np.exp(r))
        pred_close_h.append(np.array(p_seq))
    
    return np.array(true_close_h), np.array(pred_close_h)