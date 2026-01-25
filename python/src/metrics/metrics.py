import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true, y_pred, horizons, epsilon=1e-8, verbose=True):
    num_horizons = y_true.shape[1]
    metrics = {}
    
    for h in range(num_horizons):
        y_t = y_true[:, h]
        y_p = y_pred[:, h]
        
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        mape = np.mean(np.abs((y_t - y_p) / (y_t + epsilon))) * 100
        r2 = r2_score(y_t, y_p)

        y_true_direction = y_t > 0
        y_pred_direction = y_p > 0
        da = np.mean(y_true_direction == y_pred_direction) * 100

        metrics[f"{horizons[h]}"] = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "DA": da
        }

        if verbose:
            print("-"*30)
            print(f"Horizon {horizons[h]} Metrics:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.4f}")
            print(f"  DA: {da:.2f}%")

    return metrics