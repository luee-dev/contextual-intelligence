
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    ROOT_DIR
    / "reports"
    / "metrics"
    / "model_prediction_comparison.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "reports"
    / "metrics"
    / "model_disagreement_samples.csv"
)

SAMPLE_SIZE = 15


def main():
    """Extract representative records from model agreement groups."""

    df = pd.read_csv(INPUT_PATH)

    tfidf_correct = (
        df["tfidf_prediction"] == df["actual_label"]
    )

    embedding_correct = (
        df["embedding_prediction"] == df["actual_label"]
    )

    conditions = [
        tfidf_correct & embedding_correct,
        tfidf_correct & ~embedding_correct,
        ~tfidf_correct & embedding_correct,
        ~tfidf_correct & ~embedding_correct,
    ]

    group_names = [
        "both_correct",
        "tfidf_only_correct",
        "embedding_only_correct",
        "both_wrong",
    ]

    df["comparison_group"] = "unclassified"

    for condition, group_name in zip(conditions, group_names):
        df.loc[condition, "comparison_group"] = group_name

    print("\n--- MODEL AGREEMENT SUMMARY ---")

    print(
        df["comparison_group"]
        .value_counts()
        .reindex(group_names, fill_value=0)
    )

    samples = []

    for group_name in group_names:
        group = df[df["comparison_group"] == group_name]

        sample = group.sample(
            n=min(SAMPLE_SIZE, len(group)),
            random_state=42,
        )

        samples.append(sample)

    review_df = pd.concat(samples, ignore_index=True)

    review_columns = [
        "text",
        "actual_category",
        "tfidf_category",
        "embedding_category",
        "tfidf_confidence",
        "embedding_confidence",
        "comparison_group",
    ]

    review_df[review_columns].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n--- SAMPLE SELECTION ---")

    print(
        review_df["comparison_group"]
        .value_counts()
        .reindex(group_names, fill_value=0)
    )

    print("\n--- SAVING COMPLETED ---")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
