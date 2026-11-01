from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.data.clean_h2 import load_and_clean_h2
from src.data.split_h2 import split_h2
from src.evaluation.regression_metrics import (
    calculate_regression_metrics,
)
from src.preprocessing.regression import (
    FEATURES,
    TARGET,
    build_regression_preprocessor,
)


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "H2.csv"

    df = load_and_clean_h2(file_path)
    train_df, validation_df, _ = split_h2(df)

    # Select inputs and separate the ADR target.
    X_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET].to_numpy(dtype=float)

    X_validation = validation_df[FEATURES].copy()
    y_validation = validation_df[TARGET].to_numpy(
        dtype=float
    )

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_regression_preprocessor(),
                ),
                ("model", LinearRegression()),
            ]
        ),
    }

    results = []

    for model_name, model in models.items():
        # Fit once, using training data only.
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start

        # Evaluate training and validation predictions.
        evaluation_sets = [
            ("train", X_train, y_train),
            ("validation", X_validation, y_validation),
        ]

        for split_name, X, y in evaluation_sets:
            predictions = model.predict(X)

            metrics = calculate_regression_metrics(
                y, predictions
            )

            row = {
                "model": model_name,
                "split": split_name,
                "fit_seconds": fit_seconds,
            }
            row.update(metrics)
            results.append(row)

    results_df = pd.DataFrame(results)

    display_columns = [
        "model", "split", "MAE", "RMSE", "R2"
    ]
    print(
        results_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    output_dir = project_root / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "regression_baselines.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()