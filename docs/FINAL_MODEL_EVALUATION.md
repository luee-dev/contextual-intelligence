
# Final Model Evaluation

## 1. Overview

This document presents the final evaluation of three news classification models developed for the Contextual Intelligence project.

The evaluation uses the original AG News test dataset containing 7,600 records. This dataset was kept separate from model training and validation.

The models evaluated are:

1. TF-IDF with Logistic Regression
2. Sentence embeddings (all-MiniLM-L6-v2) with Logistic Regression
3. Soft-voting ensemble combining both classifiers

## 2. Evaluation Methodology

All three models were evaluated on the same 7,600 test records.

The ensemble combines predicted class probabilities using the selected validation configuration:

- TF-IDF: 70%
- MiniLM: 30%

The ensemble configuration was selected using validation macro F1 before final evaluation.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score

## 3. Overall Performance

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF | 91.53% | 91.51% | 91.51% |
| MiniLM | 89.39% | 89.38% | 89.38% |
| Soft-voting Ensemble | 91.80% | 91.78% | 91.78% |

The soft-voting ensemble achieved the strongest overall performance.

Compared with the TF-IDF baseline, the ensemble improved accuracy and macro F1 by approximately 0.27 percentage points.

## 4. Category-Level Performance

The final ensemble achieved the following classification results:

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| World | 93% | 91% | 92% |
| Sports | 96% | 98% | 97% |
| Business | 89% | 88% | 88% |
| Sci/Tech | 89% | 90% | 90% |

Sports was the strongest classification category, while Business remained the most challenging.

## 5. Validation and Test Comparison

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 92.40% | 91.80% |
| Macro F1 | 92.38% | 91.78% |

The relatively small performance difference suggests that the selected ensemble generalises reasonably well to unseen records.

## 6. Key Findings

1. TF-IDF provided the strongest standalone classification performance.
2. MiniLM embeddings contributed complementary predictive information.
3. Combining both models improved final test performance.
4. The ensemble achieved 91.80% accuracy and 91.78% macro F1.
5. Business and Sci/Tech remain the most challenging category boundaries.

## 7. Limitations

- AG News contains short news headlines and descriptions rather than complete articles.
- The four-category classification task is a simplified representation of contextual understanding.
- The dataset does not directly represent advertising suitability, brand safety, or contextual advertising outcomes.
- The ensemble weights were selected using the validation dataset.
- The results should not be interpreted as evidence of production advertising performance.

## 8. Final Model Selection

The 70/30 TF-IDF and MiniLM soft-voting ensemble was selected as the final classifier because it achieved the highest macro F1 among the evaluated models on the held-out test set.

Final test metrics:

- Accuracy: 91.80%
- Macro F1: 91.78%
- Weighted F1: 91.78%

The selected model provides the classification foundation for subsequent contextual matching and advertising analytics experiments.

## 9. Reproducibility

Evaluation script:

`src/contextual_intelligence/classification/evaluate_final.py`

Metrics:

`reports/metrics/final_test_evaluation.json`

Classification report:

`reports/metrics/final_ensemble_classification_report.csv`

Confusion matrix:

`reports/figures/final_ensemble_confusion_matrix.png`
