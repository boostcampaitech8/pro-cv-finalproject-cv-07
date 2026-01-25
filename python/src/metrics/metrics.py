import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true, y_pred, epsilon=1e-8, verbose=True):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    r2 = r2_score(y_true, y_pred)

    y_true_direction = y_true > 0
    y_pred_direction = y_pred > 0
    da = np.mean(y_true_direction == y_pred_direction) * 100

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "DA": da
    }

    if verbose:
        print("Validation Metrics:")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        print(f"Directional Accuracy (DA): {da:.2f}%")

    return metrics