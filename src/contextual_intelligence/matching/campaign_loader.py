
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]

CAMPAIGNS_PATH = ROOT_DIR / "data" / "external" / "campaigns.csv"

VALID_CATEGORIES = {
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
}

LIST_COLUMNS = [
    "target_categories",
    "target_keywords",
    "excluded_keywords",
    "excluded_categories",
]

REQUIRED_COLUMNS = {
    "campaign_id",
    "advertiser",
    "campaign_name",
    "campaign_description",
    "target_categories",
    "target_keywords",
    "excluded_keywords",
    "excluded_categories",
    "minimum_relevance_score",
    "active",
}


def parse_list(value, field_name):
    """Convert semicolon-separated CSV values into Python lists."""

    if pd.isna(value) or not str(value).strip():
        return []

    items = [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]

    if field_name in {"target_categories", "excluded_categories"}:
        invalid = set(items) - VALID_CATEGORIES

        if invalid:
            raise ValueError(
                f"Invalid categories in {field_name}: {sorted(invalid)}"
            )

    elif field_name in {"target_keywords", "excluded_keywords"}:
        items = [item.lower() for item in items]

    return items


def parse_boolean(value):
    """Convert CSV boolean values into Python booleans."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"Invalid boolean value: {value}")


def validate_campaigns(data):
    """Validate campaign records before they enter the matching system."""

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Campaign dataset is empty.")

    if data["campaign_id"].isna().any():
        raise ValueError("Campaign IDs cannot be missing.")

    if data["campaign_id"].duplicated().any():
        raise ValueError("Duplicate campaign IDs detected.")

    text_columns = [
        "campaign_id",
        "advertiser",
        "campaign_name",
        "campaign_description",
    ]

    for column in text_columns:
        if data[column].isna().any():
            raise ValueError(f"Missing values found in {column}.")

        if data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty values found in {column}.")

    for column in ["target_categories", "target_keywords"]:
        if data[column].apply(len).eq(0).any():
            raise ValueError(
                f"Every campaign must have at least one {column}."
            )

    if not data["minimum_relevance_score"].between(0, 1).all():
        raise ValueError(
            "Minimum relevance scores must be between 0 and 1."
        )

    return data


def load_campaigns(path=CAMPAIGNS_PATH):
    """Load, parse and validate the campaign dataset."""

    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    for column in LIST_COLUMNS:
        data[column] = data[column].apply(
            lambda value: parse_list(value, column)
        )

    data["minimum_relevance_score"] = pd.to_numeric(
        data["minimum_relevance_score"],
        errors="raise",
    )

    data["active"] = data["active"].apply(parse_boolean)

    data = validate_campaigns(data)

    return data


def main():
    print("\n--- LOADING CAMPAIGNS ---")

    campaigns = load_campaigns()

    print(f"Campaign records: {len(campaigns)}")
    print(f"Columns: {len(campaigns.columns)}")

    print("\n--- CAMPAIGN SUMMARY ---")

    for _, campaign in campaigns.iterrows():
        print(
            f"{campaign['campaign_id']} | "
            f"{campaign['advertiser']} | "
            f"Categories: {campaign['target_categories']} | "
            f"Active: {campaign['active']}"
        )

    print("\n--- VALIDATION COMPLETED ---")
    print("All campaign records passed validation.")


if __name__ == "__main__":
    main()
