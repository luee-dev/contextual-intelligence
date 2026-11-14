from pathlib import Path

import pandas as pd
from contextual_intelligence.preprocessing.text_cleaner import clean_text

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def clean_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean the training dataset without modifying the original test dataset.
    Removes:
    - All training records belonging to conflicting-label text groups.
    - Repeated training texts.
    - Training records whose normalized text also appears in the test set.
    """

    original_count = len(train)
    cleaned_train = train.copy()
    cleaned_train["text"] = cleaned_train["text"].apply(clean_text)
    test = test.copy()
    test["text"] = test["text"].apply(clean_text)

    # Identify text assigned to multiple labels in training
    label_counts = cleaned_train.groupby("text")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index

    # Remove every records belonging to a conflicting text group
    conflict_mask = cleaned_train["text"].isin(conflicting_texts)
    conflicting_rows_removed = int(conflict_mask.sum())

    cleaned_train = cleaned_train.loc[~conflict_mask].copy()

    # Remove repeated complete rows
    before_deduplication = len(cleaned_train)
    cleaned_train = cleaned_train.drop_duplicates(
        subset=["text"],
        keep="first",
    )

    duplicate_rows_removed = before_deduplication - len(cleaned_train)

    # Remove training texts that also appear in the test split
    test_text = set(test["text"])

    overlap_mask = cleaned_train["text"].isin(test_text)
    overlap_rows_removed = int(overlap_mask.sum())

    cleaned_train = cleaned_train.loc[~overlap_mask].copy()

    # Reset row index
    cleaned_train = cleaned_train.reset_index(drop=True)

    # Record cleaning results
    summary = {
        "original_training_records": original_count,
        "conflicting_text_groups": len(conflicting_texts),
        "conflicting_rows_removed": conflicting_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "train_test_overlap_rows_removed": overlap_rows_removed,
        "final_training_records": len(cleaned_train),
        "testing_records": len(test),
    }

    return cleaned_train, summary


def main() -> None:
    train = pd.read_csv(RAW_DIR / "ag_news_train.csv")
    test = pd.read_csv(RAW_DIR / "ag_news_test.csv")

    cleaned_train, summary = clean_datasets(train, test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_train.to_csv(
        PROCESSED_DIR / "ag_news_train_clean.csv",
        index=False,
    )

    # Preserve the original test records
    test.to_csv(
        PROCESSED_DIR / "ag_news_test.csv",
        index=False,
    )

    print("\n--- DATA CLEANING SUMMARY ---")

    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- VALIDATION ---")
    print(
        "Remaining duplicate training texts:",
        cleaned_train["text"].duplicated().sum(),
    )
    print(
        "Remaining conflicting text groups:",
        (cleaned_train.groupby("text")["label"].nunique().gt(1).sum()),
    )
    normalized_test_texts = set(test["text"].apply(clean_text))
    print(
        "Remaining train/test overlap:",
        len(set(cleaned_train["text"]) & set(test["text"])),
    )

    print("\nCleaned datasets saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
