from pathlib import Path

import pandas as pd


def load_and_clean_h2(file_path):
    """Load H2 and apply fixed cleaning rules shared by all tasks."""
    df = pd.read_csv(
        file_path,
        keep_default_na=False,
        low_memory=False,
    )

    # Remove extra spaces from text values.
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # Preserve unknown child counts as missing.
    df["Children"] = pd.to_numeric(
        df["Children"].replace("NA", pd.NA),
        errors="raise",
    ).astype("Int64")

    # Unknown country.
    df["Country"] = df["Country"].replace("NULL", pd.NA)

    # Agent and Company retain "NULL" as a meaningful category.

    # Parse dates for auditing.
    df["ReservationStatusDate"] = pd.to_datetime(
        df["ReservationStatusDate"],
        format="%Y-%m-%d",
        errors="raise",
    )

    return df


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "H2.csv"

    df = load_and_clean_h2(file_path)

    print("Shape after shared cleaning:", df.shape)

    print("\nMissing values:")
    missing_counts = df.isna().sum()
    print(missing_counts[missing_counts > 0].to_string())

    print("\nRepeated rows:", df.duplicated().sum())


if __name__ == "__main__":
    main()