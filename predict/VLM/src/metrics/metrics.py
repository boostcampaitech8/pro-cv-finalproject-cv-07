import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true, y_pred, horizons, prev_close, epsilon=1e-8, verbose=True):
    num_horizons = y_true.shape[1]
    metrics = {}
    
    rmse_list, mae_list, mape_list, r2_list, da_list = [], [], [], [], []
    
    for h in range(num_horizons):
        y_t = y_true[:, h]
        y_p = y_pred[:, h]
        p_c = prev_close[:, h]
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        mape = np.mean(np.abs((y_t - y_p) / (y_t + epsilon))) * 100
        r2 = r2_score(y_t, y_p)
        
        true_dir = (y_t - p_c) > 0
        pred_dir = (y_p - p_c) > 0
        da = np.mean(true_dir == pred_dir) * 100

        metrics[f"{horizons[h]}"] = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "DA": da
        }

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        da_list.append(da)

        if verbose:
            print("-"*30)
            print(f"Horizon {horizons[h]} Metrics:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.4f}")
            print(f"  DA: {da:.2f}%")

    metrics["Average"] = {
        "RMSE": np.mean(rmse_list),
        "MAE": np.mean(mae_list),
        "MAPE": np.mean(mape_list),
        "R2": np.mean(r2_list),
        "DA": np.mean(da_list)
    }

    if verbose:
        print("="*30)
        print("Average Metrics Across All Horizons:")
        print(f"  RMSE: {metrics['Average']['RMSE']:.6f}")
        print(f"  MAE: {metrics['Average']['MAE']:.6f}")
        print(f"  MAPE: {metrics['Average']['MAPE']:.2f}%")
        print(f"  R²: {metrics['Average']['R2']:.4f}")
        print(f"  DA: {metrics['Average']['DA']:.2f}%")
    
    return metrics