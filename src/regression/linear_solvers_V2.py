"""linear-regression solvers for prepared numerical inputs."""

import numpy as np


def _training_arrays(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or min(X.shape) < 1:
        raise ValueError("X must be a nonempty two-dimensional array.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must contain one target for each row of X.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training inputs and targets must be finite.")

    return X, y


def fit_gradient_descent(
    X, y, learning_rate=0.5, max_iterations=20000, tolerance=1e-8,
    momentum=0.0,
):
    """Minimize J = mean(squared errors) / 2 using full-batch updates.

    The comparison script standardizes all prepared columns before fitting.
    Set momentum=0 for plain GD, or use a coefficient between 0 and 1 to
    accumulate gradients in a velocity. The comparison uses momentum=0.9.
    """
    X, y = _training_arrays(X, y)

    if (
        not np.isfinite([learning_rate, tolerance, momentum]).all()
        or learning_rate <= 0
        or tolerance <= 0
        or not 0 <= momentum < 1
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError(
            "Use positive finite learning rate and tolerance, momentum in [0, 1), "
            "and a positive integer iteration limit."
        )

    weights = np.zeros(X.shape[1])
    intercept = 0.0
    velocity_weights = np.zeros(X.shape[1])
    velocity_intercept = 0.0
    cost_history = []

    # Step 0 evaluates the initial parameters before any update.
    for step in range(max_iterations + 1):
        predictions = X @ weights + intercept
        errors = predictions - y
        cost = float(np.mean(errors ** 2) / 2)

        # Compute both gradients using the same current parameters.
        gradient_weights = X.T @ errors / len(y)
        gradient_intercept = float(np.mean(errors))

        if (
            not np.isfinite(cost)
            or not np.isfinite(gradient_weights).all()
            or not np.isfinite(gradient_intercept)
        ):
            raise FloatingPointError(
                "Training diverged; check inputs and the learning rate."
            )

        cost_history.append(cost)
        largest_gradient = max(
            float(np.max(np.abs(gradient_weights))),
            abs(gradient_intercept),
        )
        converged = largest_gradient <= tolerance

        if converged or step == max_iterations:
            break

        velocity_weights = momentum * velocity_weights + gradient_weights
        velocity_intercept = momentum * velocity_intercept + gradient_intercept
        weights = weights - learning_rate * velocity_weights
        intercept = intercept - learning_rate * velocity_intercept

    return {
        "weights": weights,
        "intercept": intercept,
        "cost_history": cost_history,
        "iterations": step,
        "converged": converged,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "tolerance": tolerance,
        "largest_gradient": largest_gradient,
    }


def fit_normal_equation(X, y):
    """Solve the normal equations for a full-column-rank design matrix."""
    X, y = _training_arrays(X, y)

    # The leading column of ones supplies the intercept parameter.
    design = np.column_stack([np.ones(len(X)), X])

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            "The normal-equation design is rank deficient. "
            "Use np.linalg.lstsq on the design matrix for a general solution."
        )

    # Solve (A.T @ A) @ theta = A.T @ y without forming an inverse.
    theta = np.linalg.solve(design.T @ design, design.T @ y)

    return {
        "intercept": float(theta[0]),
        "weights": theta[1:],
    }


def predict_linear(X, fitted):
    """Predict with a fitted parameter dictionary from either solver."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(fitted["weights"]):
        raise ValueError("X has the wrong number of features.")
    if not np.isfinite(X).all():
        raise ValueError("Prediction inputs must be finite.")

    return X @ fitted["weights"] + fitted["intercept"]
