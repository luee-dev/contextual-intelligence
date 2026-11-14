from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


ROOT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT_DIR / "data" / "raw"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the ingested training and testing datasets."""

    train = pd.read_csv(RAW_DIR / "ag_news_train.csv")
    test = pd.read_csv(RAW_DIR / "ag_news_test.csv")

    return train, test


def inspect_data(train, test):
    """Print basic dataset quality and structural information."""

    print("\n--- DATASET OVERVIEW ---")
    print(f"Training records: {len(train):,}")
    print(f"Testing records: {len(test):,}")

    print("\n--- MISSING VALUES ---")
    print(train.isnull().sum())

    print("\n--- DUPLICATE RECORDS ---")
    print(f"Training duplicates: {train.duplicated().sum():,}")
    print(f"Testing duplicates: {test.duplicated().sum():,}")

    print("\n--- CATEGORY DISTRIBUTION ---")
    print(train["label_name"].value_counts())

    print("\n--- TEXT STATISTICS ---")
    print(train[["word_count", "character_count"]].describe())


def plot_category_distribution(train):
    """Visualise the distribution of article categorie."""

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=train, x="label_name", order=train["label_name"].value_counts().index
    )

    plt.title("AG News Training Dataset: Category Distribution")
    plt.xlabel("News Category")
    plt.ylabel("Number of Articles")
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "category_distribution.png", dpi=300)

    plt.close()


def plot_text_lengths(train):
    """Visualise the distribution of article text lengths."""

    plt.figure(figsize=(9, 5))

    sns.histplot(data=train, x="word_count", bins=40, kde=True)

    plt.title("Distribution of Ariclte Word counts")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "article_word_count_distribution.png", dpi=300)

    plt.close()


def plot_category_text_lengths(train):
    """Compare article lengths across categories."""

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=train,
        x="label_name",
        y="word_count",
    )

    plt.title("Article Length by News Category")
    plt.xlabel("News Category")
    plt.ylabel("Word Count")
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "word_count_by_category.png", dpi=300)

    plt.close()


def extract_frequent_terms(train):
    """Extract frequent terms for a basic vocabulary analysis."""

    vectorizer = CountVectorizer(stop_words="english", max_features=20)

    matrix = vectorizer.fit_transform(train["text"].fillna(""))

    frequencies = matrix.sum(axis=0).A1

    terms = pd.DataFrame(
        {"term": vectorizer.get_feature_names_out(), "frequency": frequencies}
    )

    terms = terms.sort_values("frequency", ascending=False)

    terms.to_csv(PROCESSED_DIR / "frequent_terms.csv", index=False)

    plt.figure(figsize=(10, 6))

    sns.barplot(data=terms, x="frequency", y="term")

    plt.title("Most Frequent Terms in AG News")
    plt.xlabel("Frequency")
    plt.ylabel("Term")
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "frequent_terms.png", dpi=300)

    plt.close()


def investigate_duplicates(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Investigate repeated text, conflicting labels and train/test overlap."""

    # Repeated text within each split
    train_text_duplicates = train.duplicated(subset=["text"]).sum()
    test_text_duplicates = test.duplicated(subset=["text"]).sum()

    # Text appearing with more that one category
    conflicting_labels = train.groupby("text")["label"].nunique().gt(1).sum()

    # Check whether any training text also appears in the test set
    train_texts = set(train["text"])
    test_texts = set(test["text"])
    overlap = train_texts.intersection(test_texts)

    print("\n--- DUPLICATE INVESTIGATION ---")
    print(f"Repeated training texts: {train_text_duplicates}")
    print(f"Repeated testing texts: {test_text_duplicates}")
    print(f"Texts with conflicting training labels: {conflicting_labels}")
    print(f"Unique texts shared between train and test: {len(overlap)}")

    if overlap:
        print("\nExample overlapping text:")
        print(next(iter(overlap)))


def inspect_label_conflicts_and_overlap(
    train: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Display records with conflicting labels and train/test overlap."""

    # Identify text group assigned to multiple labels
    label_counts = train.groupby("text")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index

    conflicts = train[train["text"].isin(conflicting_texts)][
        ["text", "label", "label_name"]
    ].sort_values(["text", "label"])

    print("\n--- CONFLICTING LABEL RECORDS ---")
    print(f"Number of conflicting text groups: {len(conflicting_texts)}")
    print(conflicts.to_string(index=False))

    # Identify text appearing in both splits
    overlap_texts = set(train["text"]).intersection(set(test["text"]))

    overlap_records = pd.concat(
        [
            train[train["text"].isin(overlap_texts)].assign(split="train"),
            test[test["text"].isin(overlap_texts)].assign(split="test"),
        ]
    )

    print("\n-- TRAIN/TEST OVERLAP RECORDS ---")
    if overlap_records.empty:
        print("No overlapping records found.")
    else:
        print(
            overlap_records[["split", "text", "label", "label_name"]].to_string(
                index=False
            )
        )

    # Save investigation results
    output_dir = ROOT_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    conflicts.to_csv(output_dir / "conflicting_label_records.csv", index=False)

    overlap_records.to_csv(output_dir / "train_test_overlap.csv", index=False)
    print("\nInvestigation files saved to: ", output_dir)


def main():
    train, test = load_data()

    inspect_data(train, test)

    plot_category_distribution(train)
    plot_text_lengths(train)
    plot_category_text_lengths(train)
    extract_frequent_terms(train)
    investigate_duplicates(train, test)
    inspect_label_conflicts_and_overlap(train, test)

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
