
import pandas as pd
import pytest

from contextual_intelligence.matching.evaluate_matching import (
    calculate_suitability_metrics,
    calculate_ranking_metrics,
)


def test_suitability_metrics():
    """Verify precision, recall and F1 for unsuitable content."""

    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 1]

    metrics = calculate_suitability_metrics(y_true, y_pred)

    assert metrics["unsuitable_precision"] == pytest.approx(2 / 3)
    assert metrics["unsuitable_recall"] == pytest.approx(2 / 3)
    assert metrics["unsuitable_f1"] == pytest.approx(2 / 3)


def test_suitability_metrics_with_no_unsuitable_predictions():
    """Verify metrics handle zero positive predictions safely."""

    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    metrics = calculate_suitability_metrics(y_true, y_pred)

    assert metrics["unsuitable_precision"] == 0.0
    assert metrics["unsuitable_recall"] == 0.0
    assert metrics["unsuitable_f1"] == 0.0


def test_ranking_metrics():
    """Verify review rate and mean reciprocal rank."""

    results = pd.DataFrame({
        "actual_decision": [
            "review",
            "suitable",
            "unsuitable",
            "review",
        ],
        "reciprocal_rank": [
            1.0,
            0.5,
            0.25,
            1.0,
        ],
    })

    metrics = calculate_ranking_metrics(results)

    assert metrics["review_rate"] == pytest.approx(0.5)
    assert metrics["mean_reciprocal_rank"] == pytest.approx(0.6875)


def test_ranking_metrics_with_no_reviews():
    """Verify review rate is zero when no decisions require review."""

    results = pd.DataFrame({
        "actual_decision": ["suitable", "unsuitable"],
        "reciprocal_rank": [1.0, 0.5],
    })

    metrics = calculate_ranking_metrics(results)

    assert metrics["review_rate"] == 0.0
    assert metrics["mean_reciprocal_rank"] == pytest.approx(0.75)
    