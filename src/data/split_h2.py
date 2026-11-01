from pathlib import Path

import pandas as pd

from src.data.clean_h2 import load_and_clean_h2


MONTH_NUMBERS = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}


def split_h2(df):
    """Split bookings into fixed arrival-date periods."""
    arrival_dates = pd.to_datetime(
        {
            "year": df["ArrivalDateYear"],
            "month": df["ArrivalDateMonth"].map(MONTH_NUMBERS),
            "day": df["ArrivalDateDayOfMonth"],
        },
        errors="raise",
    )

    if arrival_dates.isna().any():
        raise ValueError("Some bookings have missing arrival dates.")

    validation_start = pd.Timestamp("2017-01-01")
    test_start = pd.Timestamp("2017-05-01")

    train_mask = arrival_dates < validation_start

    validation_mask = (
        (arrival_dates >= validation_start)
        & (arrival_dates < test_start)
    )

    test_mask = arrival_dates >= test_start

    return (
        df.loc[train_mask].copy(),
        df.loc[validation_mask].copy(),
        df.loc[test_mask].copy(),
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "H2.csv"

    df = load_and_clean_h2(file_path)
    train_df, validation_df, test_df = split_h2(df)

    print(f"Training: {len(train_df):,} rows")
    print(f"Validation: {len(validation_df):,} rows")
    print(f"Test: {len(test_df):,} rows")


if __name__ == "__main__":
    main()