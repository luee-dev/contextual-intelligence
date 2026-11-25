
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from contextual_intelligence.matching.campaign_loader import load_campaigns
from contextual_intelligence.preprocessing.text_cleaner import clean_text


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ContextualMatcher:
    """
    Match news articles to advertising campaigns using
    semantic similarity and brand suitability rules.
    """

    def __init__(self, campaigns=None, model_name=MODEL_NAME, model=None):
        self.campaigns = (
            load_campaigns() if campaigns is None else campaigns.copy()
    )
        self.model = (
            model
            if model is not None
            else SentenceTransformer(model_name, device="cpu")
        )
        self.campaign_embeddings = self._generate_campaign_embeddings()
    
    def _generate_campaign_embeddings(self):
        """Generate semantic representations of campaign profiles."""

        campaign_profiles = []

        for _, campaign in self.campaigns.iterrows():
            profile = (
                f"{campaign['campaign_description']}. "
                f"Relevant topics: "
                f"{', '.join(campaign['target_keywords'])}."
            )

            campaign_profiles.append(profile)

        embeddings = self.model.encode(
            campaign_profiles,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )

        return embeddings

    @staticmethod
    def find_keyword_matches(text, keywords):
        """Find whole-word or whole-phrase keyword matches."""

        matches = []

        for keyword in keywords:
            escaped_keyword = re.escape(keyword.lower())

            # Allow flexible whitespace between words in phrases.
            escaped_keyword = escaped_keyword.replace(r"\ ", r"\s+")

            pattern = rf"(?<!\w){escaped_keyword}(?!\w)"

            if re.search(pattern, text):
                matches.append(keyword)

        return matches

    @staticmethod
    def evaluate_suitability(article_category, campaign, excluded_matches):
        """
        Apply campaign restrictions independently of semantic relevance.
        """

        if not campaign["active"]:
            return "unsuitable", ["Campaign is inactive."]

        if excluded_matches:
            return "unsuitable", [
                f"Excluded content detected: {', '.join(excluded_matches)}."
            ]

        if article_category in campaign["excluded_categories"]:
            return "unsuitable", [
                f"Article category '{article_category}' is excluded."
            ]

        if article_category not in campaign["target_categories"]:
            return "unsuitable", [
                f"Article category '{article_category}' is not targeted."
            ]

        return "suitable", []

    def match_article(self, article_text, article_category):
        """
        Evaluate one article against all active campaigns.

        Returns a ranked DataFrame with relevance scores,
        keyword evidence and suitability decisions.
        """

        article_text = clean_text(article_text)

        if not article_text:
            raise ValueError("Article text cannot be empty.")

        if article_category not in {
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        }:
            raise ValueError(f"Invalid article category: {article_category}")

        article_embedding = self.model.encode(
            [article_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cpu",
        )[0]

        # Embeddings are normalized, so their dot product is cosine similarity.
        similarity_scores = np.dot(
            self.campaign_embeddings,
            article_embedding,
        )

        results = []

        for index, (_, campaign) in enumerate(self.campaigns.iterrows()):

            matched_keywords = self.find_keyword_matches(
                article_text,
                campaign["target_keywords"],
            )

            excluded_matches = self.find_keyword_matches(
                article_text,
                campaign["excluded_keywords"],
            )

            relevance_score = float(similarity_scores[index])

            suitability, reasons = self.evaluate_suitability(
                article_category,
                campaign,
                excluded_matches,
            )

            if suitability == "unsuitable":
                decision = "unsuitable"

            elif relevance_score < campaign["minimum_relevance_score"]:
                decision = "review"
                reasons.append(
                    "Semantic relevance is below the campaign threshold."
                )

            else:
                decision = "suitable"
                reasons.append(
                    "Category, exclusion and relevance checks passed."
                )

            results.append(
                {
                    "campaign_id": campaign["campaign_id"],
                    "advertiser": campaign["advertiser"],
                    "campaign_name": campaign["campaign_name"],
                    "relevance_score": relevance_score,
                    "matched_keywords": matched_keywords,
                    "excluded_keywords_found": excluded_matches,
                    "suitability": suitability,
                    "decision": decision,
                    "reasons": reasons,
                }
            )

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="relevance_score",
            ascending=False,
        ).reset_index(drop=True)

        return results_df


def main():
    print("\n--- INITIALIZING CONTEXTUAL MATCHER ---")

    matcher = ContextualMatcher()

    print(f"Campaigns loaded: {len(matcher.campaigns)}")
    print(f"Campaign embedding shape: {matcher.campaign_embeddings.shape}")

    print("\n--- TESTING ARTICLE MATCHING ---")

    sample_article = (
        "A technology company has announced a new smartphone "
        "featuring artificial intelligence capabilities, "
        "improved battery life and advanced software features."
    )

    sample_category = "Sci/Tech"

    results = matcher.match_article(
        article_text=sample_article,
        article_category=sample_category,
    )

    print(f"Article category: {sample_category}")

    print("\n--- CAMPAIGN MATCHING RESULTS ---")

    for _, result in results.iterrows():
        print(
            f"\n{result['advertiser']} | "
            f"Relevance: {result['relevance_score']:.4f} | "
            f"Decision: {result['decision']}"
        )

        print(f"Matched keywords: {result['matched_keywords']}")
        print(f"Reasons: {result['reasons']}")

    print("\n--- MATCHING COMPLETED ---")


if __name__ == "__main__":
    main()
