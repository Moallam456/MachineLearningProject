"""Save the three regression report plots inside outputs/figures.

Run from the repository root:
    python -m src.evaluation.plot_regression_results

Uses the existing toy GD implementation, regression preprocessor and H2 split.
The H2 cost curves read the CSV files written by the existing comparison scripts.
The final test partition is never transformed, predicted or scored.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save files without needing a desktop plotting window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.clean_h2 import load_and_clean_h2
from src.data.split_h2 import split_h2
from src.evaluation.regression_metrics import calculate_regression_metrics
from src.preprocessing.regression import FEATURES, TARGET, build_regression_preprocessor
from src.regression.gradient_descent import fit_gradient_descent, predict_gradient_descent


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_toy(output_dir):
    # Same hypothetical examples as compare_linear.py; update both if they change.
    X = np.array([[20.0], [40.0], [60.0]])
    y = np.array([100.0, 90.0, 110.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    direct = LinearRegression().fit(X_scaled, y)
    gd = fit_gradient_descent(X_scaled, y)
    np.testing.assert_allclose(
        predict_gradient_descent(X_scaled, gd), direct.predict(X_scaled),
        atol=1e-6, rtol=0,
    )
    grid = np.linspace(10, 70, 100).reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))
    axes[0].scatter(X[:, 0], y, label="Observed", s=45)
    axes[0].plot(grid[:, 0], direct.predict(scaler.transform(grid)), label="Fitted line")
    axes[0].scatter(X[:, 0], direct.predict(X_scaled), marker="x", label="Predictions")
    axes[0].set(xlabel="Lead time", ylabel="Target", title="One feature learning example")
    axes[0].legend(fontsize=8)
    axes[1].plot(gd["cost_history"])
    axes[1].set(xlabel="GD update", ylabel="J", title="Toy gradient descent cost")
    save_figure(fig, output_dir / "toy_comparison.png")


def plot_validation(root, output_dir, metrics_dir):
    frame = load_and_clean_h2(root / "data/raw/H2.csv")
    train, validation, _ = split_h2(frame)
    model = Pipeline([
        ("preprocessor", build_regression_preprocessor()),
        ("model", LinearRegression()),
    ])
    model.fit(train[FEATURES], train[TARGET].to_numpy(dtype=float))
    actual = validation[TARGET].to_numpy(dtype=float)
    predicted = model.predict(validation[FEATURES])
    checks = calculate_regression_metrics(actual, predicted)
    # Confirm this is the same baseline experiment as the saved results.
    saved = pd.read_csv(metrics_dir / "regression_baselines.csv")
    saved = saved.loc[(saved["model"] == "linear_regression") &
                      (saved["split"] == "validation")]
    if len(saved) != 1:
        raise ValueError("Expected one linear_regression validation row in regression_baselines.csv")
    np.testing.assert_allclose(
        [checks[k] for k in ("MAE", "RMSE", "R2")],
        saved[["MAE", "RMSE", "R2"]].iloc[0].to_numpy(dtype=float),
        rtol=1e-7, atol=1e-7,
    )
    residual = actual - predicted
    checks.update({"negative_predictions": int((predicted < 0).sum()),
                   "mean_residual": float(residual.mean())})
    pd.DataFrame([checks]).to_csv(metrics_dir / "validation_plot_checks.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(actual, predicted, s=4, alpha=.12)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    axes[0].plot([low, high], [low, high], color="red", linewidth=1)
    axes[0].set(title="Validation predictions", xlabel="Actual ADR", ylabel="Predicted ADR")
    axes[1].scatter(predicted, residual, s=4, alpha=.12)
    axes[1].axhline(0, color="red", linewidth=1)
    axes[1].set(title="Validation residuals", xlabel="Predicted ADR",
                ylabel="Actual - predicted ADR")
    save_figure(fig, output_dir / "validation_diagnostics.png")
    print("Negative validation predictions:", checks["negative_predictions"])
    print(f"Mean validation residual: {checks['mean_residual']:.6f}")


def plot_h2_history(output_dir, metrics_dir):
    plain = pd.read_csv(metrics_dir / "h2_gd_cost_history.csv")
    momentum = pd.read_csv(metrics_dir / "h2_gd_momentum_cost_history.csv")
    comparison = pd.read_csv(metrics_dir / "linear_solver_comparison.csv")
    reference = comparison.loc[(comparison["solver"] == "sklearn") &
                               (comparison["split"] == "train"), "J"]
    if len(reference) != 1:
        raise ValueError("Expected one sklearn training row in linear_solver_comparison.csv")
    direct_cost = float(reference.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))
    for history, label in [(plain, "Plain GD"), (momentum, "Momentum GD")]:
        if not np.isfinite(history[["iteration", "J"]].to_numpy()).all():
            raise ValueError("Cost history contains nonfinite values")
        axes[0].plot(history["iteration"].iloc[:101], history["J"].iloc[:101], label=label)
        # Display floor only: preserve the original history CSV values.
        excess = np.maximum(history["J"] - direct_cost, 1e-10)
        axes[1].semilogy(history["iteration"], excess, label=label)
    axes[0].set(xlabel="Update", ylabel="J", title="First 100 updates")
    axes[1].set(xlabel="Update", ylabel="J above direct fit",
                title="Approach to the direct solution")
    for ax in axes:
        ax.legend(fontsize=8)
    save_figure(fig, output_dir / "h2_cost_comparison.png")


def main():
    root = Path(__file__).resolve().parents[2]
    metrics_dir = root / "outputs/metrics"
    required = ["regression_baselines.csv", "linear_solver_comparison.csv",
                "h2_gd_cost_history.csv", "h2_gd_momentum_cost_history.csv"]
    missing = [name for name in required if not (metrics_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing results: " + ", ".join(missing) + ". First run "
            "src.regression.train_baseline, src.regression.compare_h2_solvers, "
            "and src.regression.compare_h2_solvers_V2 using python -m."
        )
    output_dir = root / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_toy(output_dir)
    plot_validation(root, output_dir, metrics_dir)
    plot_h2_history(output_dir, metrics_dir)
    print("Done. All three report plots are now in outputs/figures.")


if __name__ == "__main__":
    main()
