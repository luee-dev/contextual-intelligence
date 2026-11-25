
import numpy as np
import pandas as pd
import pytest

from contextual_intelligence.matching.engine import ContextualMatcher


class FakeSentenceModel:
    """Deterministic replacement for the Sentence Transformer."""

    def encode(self, texts, **kwargs):
        vectors = []

        for text in texts:
            text = text.lower()
            vector = np.zeros(384, dtype=np.float32)

            if "tech campaign" in text or "on-topic" in text:
                vector[0] = 1.0
            elif "business campaign" in text:
                vector[1] = 1.0
            else:
                vector[2] = 1.0

            vectors.append(vector)

        return np.vstack(vectors)


@pytest.fixture
def campaigns():
    return pd.DataFrame([
        {
            "campaign_id": "CAMP001",
            "advertiser": "TechNova",
            "campaign_name": "Technology Campaign",
            "campaign_description": "Tech campaign",
            "target_categories": ["Sci/Tech"],
            "target_keywords": ["smartphone"],
            "excluded_keywords": ["cyberattack"],
            "excluded_categories": [],
            "minimum_relevance_score": 0.5,
            "active": True,
        },
        {
            "campaign_id": "CAMP002",
            "advertiser": "GlobalFinance",
            "campaign_name": "Finance Campaign",
            "campaign_description": "Business campaign",
            "target_categories": ["Business"],
            "target_keywords": ["investment"],
            "excluded_keywords": [],
            "excluded_categories": [],
            "minimum_relevance_score": 0.5,
            "active": True,
        },
    ])


@pytest.fixture
def matcher(campaigns):
    return ContextualMatcher(
        campaigns=campaigns,
        model=FakeSentenceModel(),
    )


def get_campaign_result(results, campaign_id):
    return results.loc[
        results["campaign_id"] == campaign_id
    ].iloc[0]


def test_relevant_article_is_suitable(matcher):
    results = matcher.match_article(
        "An on-topic smartphone technology announcement.",
        "Sci/Tech",
    )

    tech_result = get_campaign_result(results, "CAMP001")

    assert tech_result["relevance_score"] == pytest.approx(1.0)
    assert tech_result["suitability"] == "suitable"
    assert tech_result["decision"] == "suitable"


def test_excluded_keyword_blocks_campaign(matcher):
    results = matcher.match_article(
        "An on-topic smartphone cyberattack report.",
        "Sci/Tech",
    )

    tech_result = get_campaign_result(results, "CAMP001")

    assert tech_result["excluded_keywords_found"] == ["cyberattack"]
    assert tech_result["suitability"] == "unsuitable"
    assert tech_result["decision"] == "unsuitable"


def test_category_mismatch_blocks_campaign(matcher):
    results = matcher.match_article(
        "An on-topic smartphone technology announcement.",
        "Business",
    )

    tech_result = get_campaign_result(results, "CAMP001")

    assert tech_result["suitability"] == "unsuitable"
    assert tech_result["decision"] == "unsuitable"


def test_low_relevance_requires_review(matcher):
    results = matcher.match_article(
        "An unrelated article about something else.",
        "Sci/Tech",
    )

    tech_result = get_campaign_result(results, "CAMP001")

    assert tech_result["relevance_score"] == pytest.approx(0.0)
    assert tech_result["suitability"] == "suitable"
    assert tech_result["decision"] == "review"


def test_inactive_campaign_is_unsuitable(campaigns):
    campaigns.loc[0, "active"] = False

    matcher = ContextualMatcher(
        campaigns=campaigns,
        model=FakeSentenceModel(),
    )

    results = matcher.match_article(
        "An on-topic smartphone technology announcement.",
        "Sci/Tech",
    )

    tech_result = get_campaign_result(results, "CAMP001")

    assert tech_result["suitability"] == "unsuitable"
    assert tech_result["decision"] == "unsuitable"


def test_empty_article_text_raises_error(matcher):
    with pytest.raises(ValueError, match="cannot be empty"):
        matcher.match_article("", "Sci/Tech")


def test_invalid_article_category_raises_error(matcher):
    with pytest.raises(ValueError, match="Invalid article category"):
        matcher.match_article(
            "An on-topic smartphone announcement.",
            "Technology",
        )
