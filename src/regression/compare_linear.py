import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.regression_metrics import calculate_regression_metrics
from src.regression.gradient_descent import (
    fit_gradient_descent,
    predict_gradient_descent,
)


def main():
    # Three training examples: one feature per row.
    X_train = np.array([[20.0], [40.0], [60.0]])
    y_train = np.array([100.0, 90.0, 110.0])

    # Scale the input; keep ADR in its original units.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train the reference model and predict the training examples.
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    predictions = model.predict(X_scaled)

    # Express the fitted coefficients in the original lead-time units.
    raw_slope = model.coef_[0] / scaler.scale_[0]
    raw_intercept = model.intercept_ - raw_slope * scaler.mean_[0]

    print(f"Intercept in original units: {raw_intercept:.6f}")
    print(f"Slope in original units: {raw_slope:.6f}")
    print("Predictions:", np.round(predictions, 6))
    print("Training metrics:")

    metrics = calculate_regression_metrics(y_train, predictions)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    # Train our implementation on exactly the same prepared inputs.
    gd = fit_gradient_descent(X_scaled, y_train)
    gd_predictions = predict_gradient_descent(X_scaled, gd)

    # Convert the learned coefficients to original lead-time units.
    gd_raw_slope = gd["theta1"] / scaler.scale_[0]
    gd_raw_intercept = (
        gd["theta0"] - gd_raw_slope * scaler.mean_[0]
    )

    print("\nGradient descent:")
    print(f"Intercept in original units: {gd_raw_intercept:.6f}")
    print(f"Slope in original units: {gd_raw_slope:.6f}")
    print("Predictions:", np.round(gd_predictions, 6))
    print(f"Updates: {gd['iterations']}")
    print(f"Converged: {gd['converged']}")
    print(f"Initial J: {gd['cost_history'][0]:.6f}")
    print(f"Final J: {gd['cost_history'][-1]:.6f}")

    gd_metrics = calculate_regression_metrics(y_train, gd_predictions)

    print("\nTraining metric comparison:")
    for name in metrics:
        print(
            f"{name}: sklearn={metrics[name]:.6f}, "
            f"GD={gd_metrics[name]:.6f}"
        )

    # Check agreement within an absolute tolerance of 0.000001.
    np.testing.assert_allclose(
        gd_predictions, predictions, rtol=0, atol=1e-6
    )
    print("Prediction comparison passed.")
    
if __name__ == "__main__":
    main()