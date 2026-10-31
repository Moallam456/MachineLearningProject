from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true, y_pred):
    """Calculate metrics for a single regression target."""
    mse = float(mean_squared_error(y_true, y_pred))

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": float(r2_score(y_true, y_pred)),
        "J": mse / 2,
    }