
import pandas as pd
import pytest

from contextual_intelligence.matching.campaign_loader import (
    load_campaigns,
    parse_boolean,
    parse_list,
    validate_campaigns,
)


def test_load_campaigns():
    campaigns = load_campaigns()

    assert len(campaigns) == 8
    assert campaigns["campaign_id"].is_unique
    assert campaigns["active"].all()


def test_list_fields_are_parsed():
    campaigns = load_campaigns()

    campaign = campaigns.loc[
        campaigns["campaign_id"] == "CAMP002"
    ].iloc[0]

    assert campaign["target_categories"] == ["Business", "Sci/Tech"]
    assert "renewable energy" in campaign["target_keywords"]


def test_keywords_are_lowercase():
    result = parse_list(
        "Artificial Intelligence;SOLAR POWER",
        "target_keywords",
    )

    assert result == ["artificial intelligence", "solar power"]


def test_empty_list_value():
    assert parse_list("", "target_keywords") == []


def test_invalid_category():
    with pytest.raises(ValueError, match="Invalid categories"):
        parse_list("Technology", "target_categories")


def test_boolean_parsing():
    assert parse_boolean("true") is True
    assert parse_boolean("false") is False


def test_invalid_boolean():
    with pytest.raises(ValueError, match="Invalid boolean"):
        parse_boolean("yes")


def test_duplicate_campaign_ids():
    campaigns = load_campaigns()

    duplicated = pd.concat(
        [campaigns, campaigns.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate campaign IDs"):
        validate_campaigns(duplicated)


def test_invalid_relevance_score():
    campaigns = load_campaigns()
    campaigns.loc[0, "minimum_relevance_score"] = 1.5

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_campaigns(campaigns)
