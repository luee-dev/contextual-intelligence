from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

ERRORS_PATH = (
    ROOT_DIR
    / "reports"
    / "metrics"
    / "misclassified_records.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "reports"
    / "metrics"
    / "error_samples_for_review.csv"
)

def main():
    if not ERRORS_PATH.exists():
        raise FileNotFoundError(
            f"Misclassified records not found: {ERRORS_PATH}"
        )

    errors = pd.read_csv(ERRORS_PATH)

    # Focus on the two most problematic directions.
    pairs = [
        ("Business", "Sci/Tech"),
        ("Sci/Tech", "Business"),
        ("World", "Business"),
    ]

    samples = []

    for actual, predicted in pairs:
        subset = errors.loc[
            (errors["actual_category"] == actual)
            & (errors["predicted_category"] == predicted)
        ].copy()

        subset = subset.sort_values(
            "prediction_confidence",
            ascending=False,
        )

        selected = subset.head(5).copy()
        samples.append(selected)

        print(f"\n{'=' * 75}")
        print(f"ACTUAL: {actual} | PREDICTED: {predicted}")
        print(f"{'=' * 75}")

        for index, row in selected.iterrows():
            print(
                f"\nConfidence: "
                f"{row['prediction_confidence']:.2%}"
            )
            print(f"Text: {row['text']}")

    if samples:
        review_df = pd.concat(samples, ignore_index=True)
        review_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nReview samples saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()