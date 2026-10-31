import numpy as np


def fit_gradient_descent(
    X, y, learning_rate=0.1, max_iterations=10000, tolerance=1e-8
):
    """Fit one-feature linear regression using full-batch gradient descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("This version expects X with shape (samples, 1).")
    if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
        raise ValueError("y must contain one target for each sample.")
    if learning_rate <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError(
            "Use positive learning rate, iteration limit and tolerance."
        )

    x = X[:, 0]
    theta0 = 0.0
    theta1 = 0.0
    cost_history = []

    # Step 0 evaluates the initial model.
    # Each later step evaluates the model after an update.
    for step in range(max_iterations + 1):
        predictions = theta0 + theta1 * x
        errors = predictions - y
        cost = float(np.mean(errors ** 2) / 2)

        # Compute BOTH gradients before updating either parameter.
        gradient0 = float(np.mean(errors))
        gradient1 = float(np.mean(errors * x))

        if not np.all(np.isfinite([cost, gradient0, gradient1])):
            raise FloatingPointError(
                "Nonfinite training values; check inputs and learning rate."
            )

        cost_history.append(cost)

        converged = max(abs(gradient0), abs(gradient1)) <= tolerance

        if converged or step == max_iterations:
            break

        theta0 = theta0 - learning_rate * gradient0
        theta1 = theta1 - learning_rate * gradient1

    return {
        "theta0": theta0,
        "theta1": theta1,
        "cost_history": cost_history,
        "iterations": step,
        "converged": converged,
    }


def predict_gradient_descent(X, fitted):
    """Predict using inputs transformed by the same fitted scaler."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("This version expects X with shape (samples, 1).")

    return fitted["theta0"] + fitted["theta1"] * X[:, 0]