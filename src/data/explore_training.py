from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.clean_h2 import load_and_clean_h2
from src.data.split_h2 import split_h2


def main():
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / "raw" / "H2.csv"

    df = load_and_clean_h2(file_path)
    train_df = split_h2(df)[0]

    metrics_dir = project_root / "outputs" / "metrics"
    figures_dir = project_root / "outputs" / "figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Summarize numerical columns using training data only.
    summary = (
        train_df.select_dtypes(include="number")
        .describe(percentiles=[0.5, 0.95, 0.99])
        .T
    )
    summary.to_csv(metrics_dir / "training_summary.csv")

    total_nights = (
        train_df["StaysInWeekendNights"]
        + train_df["StaysInWeekNights"]
    )

    # Require all three counts when calculating total guests.
    total_guests = train_df[
        ["Adults", "Children", "Babies"]
    ].sum(axis=1, min_count=3)

    checks = pd.Series(
        {
            "missing_children": train_df["Children"].isna().sum(),
            "missing_country": train_df["Country"].isna().sum(),
            "zero_adr": train_df["ADR"].eq(0).sum(),
            "zero_nights": total_nights.eq(0).sum(),
            "zero_total_guests": total_guests.eq(0).sum(),
            "adr_above_1000_review_only": (
                train_df["ADR"].gt(1000).sum()
            ),
            "repeated_rows": train_df.duplicated().sum(),
        },
        name="count",
    )

    checks.to_csv(
        metrics_dir / "training_quality_checks.csv",
        index_label="check",
    )

    # Show the full distribution and a closer view.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(train_df["ADR"], bins=60, log=True)
    axes[0].set_title("Full ADR range (log counts)")
    axes[0].set_xlabel("Recorded ADR")
    axes[0].set_ylabel("Bookings (log scale)")

    cutoff = train_df["ADR"].quantile(0.99)
    central_adr = train_df.loc[
        train_df["ADR"] <= cutoff, "ADR"
    ]

    axes[1].hist(central_adr, bins=50)
    axes[1].set_title(
        f"ADR up to the 99th percentile ({cutoff:.2f})"
    )
    axes[1].set_xlabel("Recorded ADR")
    axes[1].set_ylabel("Bookings")

    fig.text(
        0.5,
        0.01,
        "Right panel shows a subset for visibility; "
        "the training data is unchanged.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(
        figures_dir / "training_adr_distribution.png",
        dpi=150,
    )
    plt.close(fig)

    print("Training rows:", len(train_df))
    print("\nQuality checks:")
    print(checks.to_string())
    print("\nSaved two CSV reports and the ADR figure.")


if __name__ == "__main__":
    main()