from pathlib import Path

import pandas as pd


def main():
    # Locate the project root, regardless of the terminal's location.
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "H2.csv"

    df = pd.read_csv(
        file_path,
        keep_default_na=False,
        low_memory=False,
    )

    print("Dataset shape:", df.shape)

    print("\nColumn data types:")
    print(df.dtypes.to_string())

    print("\nText markers to investigate:")
    markers = ["", "NA", "N/A", "NULL", "NaN", "nan", "None"]

    for column in df.columns:
        text = df[column].astype("string").str.strip()
        counts = text[text.isin(markers)].value_counts()

        if not counts.empty:
            print(f"{column}: {counts.to_dict()}")

    print("\nRows repeating an earlier row:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    main()