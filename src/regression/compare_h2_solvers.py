"""Compare three fitting methods for the same linear model."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.clean_h2 import load_and_clean_h2
from src.data.split_h2 import split_h2
from src.evaluation.regression_metrics import calculate_regression_metrics
from src.preprocessing.regression import (
    FEATURES,
    TARGET,
    build_regression_preprocessor,
)
from src.regression.linear_solvers import (
    fit_gradient_descent,
    fit_normal_equation,
    predict_linear,
)


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "H2.csv"
    df = load_and_clean_h2(file_path)
    train_df, validation_df, _ = split_h2(df)

    # This comparison additionally standardizes the category indicators.
    # Every solver receives exactly the same transformed inputs and targets.
    preparation = Pipeline(
        steps=[
            ("columns", build_regression_preprocessor()),
            ("solver_scaler", StandardScaler()),
        ]
    )
    X_train = preparation.fit_transform(train_df[FEATURES])
    X_validation = preparation.transform(validation_df[FEATURES])
    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_validation = validation_df[TARGET].to_numpy(dtype=float)

    print("Prepared training shape:", X_train.shape)
    print("Prepared validation shape:", X_validation.shape)

    fitted_models = {}
    fit_times = {}

    start = perf_counter()
    fitted_models["sklearn"] = LinearRegression().fit(X_train, y_train)
    fit_times["sklearn"] = perf_counter() - start

    for name, fit_function in [
        ("normal_equation", fit_normal_equation),
        ("gradient_descent", fit_gradient_descent),
    ]:
        start = perf_counter()
        fitted_models[name] = fit_function(X_train, y_train)
        fit_times[name] = perf_counter() - start

    gd = fitted_models["gradient_descent"]
    print("\nGD updates:", gd["iterations"])
    print("GD converged:", gd["converged"])
    print(f"GD initial J: {gd['cost_history'][0]:.6f}")
    print(f"GD final J: {gd['cost_history'][-1]:.6f}")

    if not gd["converged"]:
        raise RuntimeError(
            "GD reached its iteration limit. Check the learning rate, "
            "input scaling, and iteration limit before comparing solutions."
        )

    reference = fitted_models["sklearn"].predict(X_validation)
    results = []

    for name, fitted in fitted_models.items():
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("validation", X_validation, y_validation),
        ]:
            if name == "sklearn":
                predictions = fitted.predict(X)
            else:
                predictions = predict_linear(X, fitted)

            metrics = calculate_regression_metrics(y, predictions)
            row = {
                "solver": name,
                "split": split_name,
                "solver_fit_seconds": fit_times[name],
            }
            row.update(metrics)
            results.append(row)

            if split_name == "validation":
                gap = float(np.max(np.abs(predictions - reference)))
                print(f"{name}: maximum validation prediction gap = {gap:.3e}")
                np.testing.assert_allclose(
                    predictions, reference, rtol=0, atol=1e-5
                )

    results_df = pd.DataFrame(results)
    validation_results = results_df.loc[
        results_df["split"] == "validation",
        ["solver", "MAE", "RMSE", "R2"],
    ]
    print("\nValidation comparison:")
    print(validation_results.round(6).to_string(index=False))
    print("\nPrediction comparison passed.")

    output_dir = project_root / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "linear_solver_comparison.csv", index=False)
    pd.DataFrame(
        {
            "iteration": range(len(gd["cost_history"])),
            "J": gd["cost_history"],
        }
    ).to_csv(output_dir / "h2_gd_cost_history.csv", index=False)
    print(f"Saved comparison and GD history to: {output_dir}")


if __name__ == "__main__":
    main()
