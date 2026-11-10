from contextual_intelligence.data.ingest import (
    load_ag_news,
    prepare_dataset
)

def test_ag_news_loading():
    train_df, test_df = load_ag_news()

    assert len(train_df) == 120000
    assert len(test_df) == 7600


def test_dataset_preparation():
    train_df, _ = load_ag_news()

    prepared = prepare_dataset(train_df.head(10))

    assert "label_name" in prepared.columns
    assert "word_count" in prepared.columns
    assert "character_count" in prepared.columns

    assert prepared["label_name"].notna().all()
    assert prepared["word_count"].notna().all()
