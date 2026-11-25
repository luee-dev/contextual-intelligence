
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from contextual_intelligence.matching.engine import ContextualMatcher


ROOT_DIR = Path(__file__).resolve().parents[3]

EVALUATION_DATA_PATH = (
    ROOT_DIR / "data" / "external" / "matching_evaluation.csv"
)

METRICS_DIR = ROOT_DIR / "reports" / "metrics"

REQUIRED_COLUMNS = {
    "article_id",
    "article_text",
    "article_category",
    "expected_campaign_id",
    "expected_decision",
    "scenario",
}

def calculate_suitability_metrics(y_true, y_pred):
    """Calculate metrics for detecting unsuitable content."""

    return {
        "unsuitable_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "unsuitable_recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "unsuitable_f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }


def calculate_ranking_metrics(results_df):
    """Calculate campaign ranking and review metrics."""

    return {
        "review_rate": float(
            results_df["actual_decision"].eq("review").mean()
        ),
        "mean_reciprocal_rank": float(
            results_df["reciprocal_rank"].mean()
        ),
    }

def load_evaluation_data(path=EVALUATION_DATA_PATH):
    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Evaluation dataset is empty.")

    if data["article_id"].duplicated().any():
        raise ValueError("Duplicate article IDs detected.")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Missing values found in evaluation dataset.")

    return data


def evaluate_matching():
    print("\n--- LOADING EVALUATION DATA ---")

    articles = load_evaluation_data()

    print(f"Evaluation articles: {len(articles)}")

    print("\n--- INITIALIZING MATCHER ---")

    matcher = ContextualMatcher()

    print(f"Campaigns loaded: {len(matcher.campaigns)}")

    evaluation_results = []
    ranking_results = []
    suitability_true = []
    suitability_predicted = []

    print("\n--- EVALUATING ARTICLES ---")

    for _, article in articles.iterrows():

        matches = matcher.match_article(
            article_text=article["article_text"],
            article_category=article["article_category"],
        )

        expected_campaign_id = article["expected_campaign_id"]

        expected_match = matches.loc[
            matches["campaign_id"] == expected_campaign_id
        ]

        if expected_match.empty:
            raise ValueError(
                f"Expected campaign {expected_campaign_id} "
                f"not found for article {article['article_id']}."
            )

        expected_match = expected_match.iloc[0]

        top_match = matches.iloc[0]

        expected_rank = int(
            matches.index.get_loc(expected_match.name) + 1
        )

        reciprocal_rank = 1.0 / expected_rank

        decision_correct = (
            expected_match["decision"] == article["expected_decision"]
        )

        suitability_true.append(
            article["expected_decision"] == "unsuitable"
        )

        suitability_predicted.append(
            expected_match["decision"] == "unsuitable"
        )

        top_campaign_correct = (
            top_match["campaign_id"] == expected_campaign_id
        )

        evaluation_results.append({
            "article_id": article["article_id"],
            "scenario": article["scenario"],
            "expected_campaign_id": expected_campaign_id,
            "expected_decision": article["expected_decision"],
            "actual_decision": expected_match["decision"],
            "decision_correct": bool(decision_correct),
            "top_campaign_id": top_match["campaign_id"],
            "top_campaign_correct": bool(top_campaign_correct),
            "relevance_score": float(
                expected_match["relevance_score"]
            ),
            "matched_keywords": expected_match["matched_keywords"],
            "excluded_keywords_found": (
                expected_match["excluded_keywords_found"]
            ),
            "expected_campaign_rank": expected_rank,
            "reciprocal_rank": reciprocal_rank,
        })

        for rank, (_, result) in enumerate(
            matches.iterrows(), start=1
        ):
            ranking_results.append({
                "article_id": article["article_id"],
                "campaign_id": result["campaign_id"],
                "rank": rank,
                "relevance_score": float(
                    result["relevance_score"]
                ),
                "decision": result["decision"],
                "suitability": result["suitability"],
            })

    results_df = pd.DataFrame(evaluation_results)
    rankings_df = pd.DataFrame(ranking_results)

    decision_accuracy = results_df["decision_correct"].mean()

    top_campaign_accuracy = (
        results_df["top_campaign_correct"].mean()
    )

    mean_reciprocal_rank = results_df["reciprocal_rank"].mean()

    review_rate = (
        results_df["actual_decision"].eq("review").mean()
    )

    # unsuitable_precision = precision_score(
    #     suitability_true,
    #     suitability_predicted,
    #     zero_division=0,
    # )

    # unsuitable_recall = recall_score(
    #     suitability_true,
    #     suitability_predicted,
    #     zero_division=0,
    # )

    # unsuitable_f1 = f1_score(
    #     suitability_true,
    #     suitability_predicted,
    #     zero_division=0,
    # )

    suitability_metrics = calculate_suitability_metrics(
        suitability_true,
        suitability_predicted,
    )

    ranking_metrics = calculate_ranking_metrics(results_df)

    summary = {
        "evaluation_articles": int(len(results_df)),
        "campaigns_evaluated": int(len(matcher.campaigns)),
        "decision_accuracy": float(decision_accuracy),
        "top_campaign_accuracy": float(top_campaign_accuracy),
        **suitability_metrics,
        **ranking_metrics,
        "correct_decisions": int(
            results_df["decision_correct"].sum()
        ),
        "correct_top_campaigns": int(
            results_df["top_campaign_correct"].sum()
        ),
        "review_rate": float(review_rate),
        "mean_reciprocal_rank": float(mean_reciprocal_rank),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        METRICS_DIR / "matching_evaluation_results.csv",
        index=False,
    )

    rankings_df.to_csv(
        METRICS_DIR / "campaign_ranking_results.csv",
        index=False,
    )

    with open(
        METRICS_DIR / "matching_evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n--- MATCHING EVALUATION RESULTS ---")

    print(
        results_df[
            [
                "article_id",
                "expected_campaign_id",
                "actual_decision",
                "decision_correct",
                "top_campaign_id",
                "top_campaign_correct",
            ]
        ].to_string(index=False)
    )

    print("\n--- SUMMARY ---")

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\nEvaluation reports saved successfully.")


if __name__ == "__main__":
    evaluate_matching()
