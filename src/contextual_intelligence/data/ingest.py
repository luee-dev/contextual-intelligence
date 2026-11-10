from datasets import load_dataset
import pandas as pd
from pathlib import Path


def load_ag_news() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AG News training and test dataset."""

    dataset = load_dataset("fancyzhx/ag_news")

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    return train_df, test_df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the raw AG News records for analysis."""

    df = df.copy()

    label_mapping = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }

    df["label_name"] = df["label"].map(label_mapping)

    df["text"] = (
        df["text"].fillna("").str.strip()
    )

    df["word_count"] = df["text"].str.split().str.len()

    df["character_count"] = df["text"].str.len()

    return df

def main():
    train_df, test_df = load_ag_news()

    train_df = prepare_dataset(train_df)
    test_df = prepare_dataset(test_df)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        output_dir / "ag_news_train.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "ag_news_test.csv",
        index=False
    )

    print("Dataset ingestion completed")
    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")

    print("\nTraining dataset preview:")
    print(train_df.head())

    print("\nCategory distribution:")
    print(train_df["label_name"].value_counts())


if __name__ == "__main__":
    main()
