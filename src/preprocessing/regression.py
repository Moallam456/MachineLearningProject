from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.clean_h2 import load_and_clean_h2
from src.data.split_h2 import split_h2


TARGET = "ADR"

NUMERIC_FEATURES = [
    "LeadTime",
    "Adults",
    "Children",
    "Babies",
    "StaysInWeekendNights",
    "StaysInWeekNights",
]

CATEGORICAL_FEATURES = [
    "ArrivalDateMonth",
    "ReservedRoomType",
    "Meal",
    "MarketSegment",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_regression_preprocessor():
    """Create a fresh, unfitted regression preprocessor."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),                     
        ]
    )

    categorical_encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "H2.csv"

    df = load_and_clean_h2(file_path)
    train_df, validation_df, _ = split_h2(df)

    X_train_raw = train_df[FEATURES].copy()
    X_validation_raw = validation_df[FEATURES].copy()

    preprocessor = build_regression_preprocessor()

    # Learn preprocessing statistics from training inputs.
    X_train_prepared = preprocessor.fit_transform(X_train_raw)

    # Reuse those statistics for validation inputs.
    X_validation_prepared = preprocessor.transform(
        X_validation_raw
    )

    print("Training shape:", X_train_prepared.shape)
    print("Validation shape:", X_validation_prepared.shape)

    print(
        "Training inputs finite:",
        np.isfinite(X_train_prepared).all(),
    )
    print(
        "Validation inputs finite:",
        np.isfinite(X_validation_prepared).all(),
    )


if __name__ == "__main__":
    main()