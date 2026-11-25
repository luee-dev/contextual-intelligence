# Contextual Matching Evaluation

## 1. Overview

This evaluation assesses the contextual advertising matching engine developed as part of the Contextual Intelligence project.

The system evaluates article content against simulated advertising campaigns using sentence embeddings, cosine similarity and explicit brand suitability rules.

The objective is to determine whether the system can identify relevant campaigns, rank them and apply campaign-specific suitability decisions.

## 2. Evaluation Dataset

A manually constructed evaluation dataset was created containing:

- 10 representative news articles.
- 8 simulated advertising campaigns.
- 6 evaluation fields, including article text, category, expected campaign, expected decision and scenario.

The evaluation scenarios cover:

- Strong semantic matches.
- Semantic matches using different wording.
- Excluded content.
- Category-specific matching.
- Cross-category matching.

The dataset is intended for functional and exploratory evaluation rather than statistically representative performance measurement.

## 3. Evaluation Methodology

Each article was evaluated against all eight advertising campaigns.

The matching engine:

1. Generates an embedding representation of the article.
2. Calculates cosine similarity against campaign embeddings.
3. Identifies matching and excluded keywords.
4. Checks campaign status and category restrictions.
5. Applies campaign-specific relevance thresholds.
6. Produces suitability decisions and ranked campaign results.

The evaluation compares the actual decision for each expected campaign against its manually assigned expected decision.

It also checks whether the expected campaign appears first in the ranking.

## 4. Evaluation Results

The matching engine was evaluated against 10 manually constructed articles and 8 simulated advertising campaigns, producing 80 article-campaign comparisons.

| Metric | Result |
|---|---:|
| Evaluation articles | 10 |
| Campaigns evaluated | 8 |
| Article-campaign comparisons | 80 |
| Decision agreement | 100% |
| Top campaign accuracy | 100% |
| Unsuitable precision | 1.00 |
| Unsuitable recall | 1.00 |
| Unsuitable F1-score | 1.00 |
| Review rate | 10% |
| Mean Reciprocal Rank (MRR) | 1.00 |

### Interpretation

- **Decision agreement:** All 10 expected-campaign decisions matched the manually specified outcomes.
- **Top campaign accuracy:** The expected campaign ranked first for all 10 articles.
- **Unsuitable precision and recall:** All three expected unsuitable cases were correctly identified, with no false positives in this sample.
- **Review rate:** One of the 10 expected-campaign decisions was sent for manual review.
- **Mean Reciprocal Rank:** Every expected campaign ranked first, producing an MRR of 1.00.

These results demonstrate functional consistency on the constructed evaluation scenarios. They should not be interpreted as estimates of real-world performance.

## 5. Key Findings

### Semantic relevance

The embedding-based approach successfully ranked the intended campaigns for all ten evaluation articles.

### Brand suitability

The exclusion rules correctly identified specified sensitive content, including cybersecurity incidents and financial misconduct.

### Relevance thresholds

The SportEdge campaign example demonstrated that the highest-ranked campaign may still require manual review when its relevance score falls below the configured threshold.

### Category restrictions

Campaign category restrictions prevented placements that did not satisfy campaign targeting requirements.

## 6. Limitations

- The evaluation dataset contains only ten manually constructed examples.
- Expected campaign assignments and decisions are manually specified rather than independently annotated by multiple reviewers.
- The evaluation does not establish generalisation to unseen real-world articles.
- Exact keyword matching may miss synonyms, indirect references, negation and contextual variations.
- Semantic similarity scores are not calibrated probabilities of campaign suitability.
- The evaluation does not yet measure precision, recall or ranking quality across a larger labelled dataset.

## 7. Generated Evaluation Reports

The evaluation process produces:

- `reports/metrics/matching_evaluation_results.csv`
- `reports/metrics/campaign_ranking_results.csv`
- `reports/metrics/matching_evaluation_summary.json`

## 8. Conclusion

The initial evaluation demonstrates that the contextual matching engine can combine semantic similarity, campaign targeting rules and brand suitability checks to produce explainable advertising recommendations.

The results provide a functional validation of the implemented approach.

Further evaluation using a larger, independently reviewed dataset is required before drawing conclusions about real-world matching performance.



## Category Restriction Analysis

### Objective

Investigate whether strict campaign category targeting rejects potentially relevant articles despite strong semantic similarity.

### Methodology

Analysed campaign matches classified as unsuitable because the predicted article category was not included in the campaign's target categories.

Identified records whose semantic relevance scores exceeded their respective campaign thresholds.

### Results

- Total category-blocked evaluations: 45,600.
- Category-blocked matches above relevance thresholds: 45.
- Potentially missed matches: approximately 0.10% of category-blocked evaluations.
- TravelSphere recorded the highest number of above-threshold blocked matches (20).
- TechNova recorded 10, SportEdge 7, HealthPlus 5, AutoMotion 2 and GlobalFinance 1.
- GreenFuture and EducationFirst recorded no above-threshold blocked matches.

### Qualitative Findings

Some blocked matches represented credible cross-category opportunities:

- Tourism articles classified as Business or Sci/Tech.
- Electric vehicle articles classified as World.
- Travel and hotel articles classified outside World.

Other matches demonstrated that semantic similarity does not necessarily establish commercial relevance.

### Decision

Retain strict category targeting as the default campaign policy.

Future development may introduce configurable strict and flexible category-targeting policies. Flexible targeting should require additional validation rather than relying exclusively on cosine similarity.

### Limitations

- The analysis uses a small set of synthetic campaign configurations.
- AG News contains headlines and short descriptions rather than complete articles.
- Semantic similarity is not a calibrated probability of advertising relevance.
- The 45 matches represent potential opportunities, not confirmed suitable placements.
