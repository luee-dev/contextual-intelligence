import pandas as pd

from contextual_intelligence.data.clean import clean_datasets


def test_cleaning_removes_conflicts_duplicates_and_overlap():
    train = pd.DataFrame(
        {
            "text": [
                "normal story",
                "normal story",
                "conflicting story",
                "conflicting story",
                "overlapping story",
            ],
            "label": [0, 0, 2, 3, 1],
            "label_name": [
                "World",
                "World",
                "Business",
                "Sci/Tech",
                "Sports",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "text": ["overlapping story"],
            "label": [1],
            "label_name": ["Sports"],
        }
    )

    cleaned, summary = clean_datasets(train, test)
    
    
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text"] == "normal story"
    assert cleaned.duplicated().sum() == 0
    assert cleaned["text"].duplicated().sum() == 0
    assert summary["conflicting_text_groups"] == 1
    assert summary["conflicting_rows_removed"] == 2
    assert summary["duplicate_rows_removed"] == 1
    assert summary["train_test_overlap_rows_removed"] == 1


def test_test_dataset_is_not_modified():
    train = pd.DataFrame(
        {
            "text": ["story"],
            "label": [0],
        }
    )

    test = pd.DataFrame(
        {
            "text": ["story"],
            "label": [0],
        }
    )

    original_test = test.copy(deep=True)

    clean_datasets(train, test)

    pd.testing.assert_frame_equal(test, original_test)
