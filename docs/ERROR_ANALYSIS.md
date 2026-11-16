
# Baseline Model Error Analysis

## 1. Overview

This analysis investigates errors made by the TF-IDF and Logistic Regression baseline on the 23,964-record validation dataset.

The objective is to identify classification weaknesses and determine whether semantic representations could improve contextual understanding.

## 2. Overall Error Statistics

| Metric | Result |
|---|---:|
| Validation records | 23,964 |
| Correct predictions | 22,055 |
| Misclassified records | 1,909 |
| Overall error rate | 7.97% |
| High-confidence errors (80% or above) | 433 |

## 3. Errors by Actual Category

| Category | Errors | Error rate |
|---|---:|---:|
| Business | 675 | 11.27% |
| Sci/Tech | 559 | 9.34% |
| World | 576 | 9.61% |
| Sports | 99 | 1.65% |

Business had the highest error rate, while Sports had the lowest.

## 4. Most Common Classification Errors

| Actual category | Predicted category | Errors |
|---|---|---:|
| Business | Sci/Tech | 455 |
| Sci/Tech | Business | 353 |
| World | Business | 217 |
| World | Sports | 180 |
| World | Sci/Tech | 179 |
| Business | World | 169 |
| Sci/Tech | World | 161 |

The three most frequent error directions account for approximately 53.7% of all misclassifications.

## 5. Qualitative Findings

### Business and Sci/Tech overlap

Articles about semiconductor development, software products, technology companies and digital services were frequently assigned to the alternative category.

This indicates that the categories share vocabulary and subject matter.

### Economic reporting and World news

International economic developments, currency movements, oil prices and financial policy were sometimes classified as Business rather than World.

This suggests that the model relies heavily on financial terminology without sufficiently representing the broader editorial context.

### Sports classification

Sports achieved a substantially lower error rate than the other categories. Its specialised vocabulary appears to provide more distinctive classification features.

## 6. High-Confidence Misclassifications

Several incorrect predictions had confidence levels above 95%, including examples exceeding 99%.

This demonstrates that a high predicted probability does not necessarily imply agreement with the dataset label.

The predicted probabilities should not be interpreted as calibrated confidence without further evaluation.

## 7. Limitations of the Analysis

- The inspected examples were selected from the highest-confidence errors in three major confusion pairs.
- The examples are not a random sample of all misclassified records.
- Some records may legitimately fit multiple editorial categories.
- Dataset labels are treated as the evaluation reference, although category boundaries can be subjective.

## 8. Implications for Model Development

The baseline relies on TF-IDF word and phrase frequencies. It does not explicitly represent contextual meaning or semantic relationships.

The findings motivate an embedding-based approach that can represent the meaning of complete news descriptions.

The next experiment will compare a pretrained sentence-embedding representation combined with a classifier against the TF-IDF baseline.

## 9. Output Artifacts

- `reports/metrics/error_analysis_summary.json`
- `reports/metrics/confusion_pairs.csv`
- `reports/metrics/misclassified_records.csv`
- `reports/metrics/error_samples_for_review.csv`
