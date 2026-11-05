"""Original Trial 0 preprocessing, shared by the Trial 1 model."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "IsCanceled"
CATEGORICAL_FEATURES = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment",
    "DistributionChannel", "ReservedRoomType", "Agent", "CustomerType",
]
NUMERIC_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_split(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns: {missing}")
    if df.empty:
        raise ValueError(f"{path.name}: no data rows")
    for column in FEATURES + [TARGET]:
        df[column] = df[column].str.strip().replace(
            {"": float("nan"), "NULL": float("nan"), "null": float("nan")}
        )
    y = pd.to_numeric(df[TARGET], errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f"{path.name}: {TARGET} must contain only 0 and 1")
    X = df[FEATURES].copy()
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")
    X["Agent"] = X["Agent"].fillna("0")
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(object).where(X[column].notna(), float("nan"))
    return X, y.astype(int)


def build_preprocessor():
    # Fitted by model.fit on training only, then reused for held-out splits.
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
