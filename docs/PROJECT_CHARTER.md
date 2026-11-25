# Project Charter: Contextual Intelligence

## 1. Project Overview

**Project name:** Contextual Intelligence

**Project domain:** Data Science, Natural Language Processing and Contextual Advertising

**Project type:** Independent portfolio project

## 2. Business Problem

Digital advertising platforms need to identify relevant advertising opportunities based on the content and context of online publications.

Keyword matching alone may fail to recognise the underlying meaning of an article. Two articles can discuss similar concepts using different vocabulary, while the same word can have different meanings depending on context.

This project investigates how Natural Language Processing (NLP), machine learning and semantic text representations can be used to classify online articles, identify contextual signals and match content with relevant advertising campaigns.

## 3. Project Objectives

1. Develop a reproducible pipeline for processing article content.
2. Build a machine learning model for article topic classification.
3. Extract semantic representations of articles and advertising campaigns.
4. Develop a contextual campaign-ranking approach.
5. Introduce brand suitability rules to identify potentially inappropriate placements.
6. Evaluate model performance using appropriate metrics and labelled examples.
7. Present results through an interactive analytical application.


## 4. Technical Approach and Methodology

The project follows an incremental data science and machine learning workflow.

### 4.1 Data Preparation and Exploration
- Load and inspect the AG News dataset.
- Perform exploratory data analysis to understand article characteristics and category distributions.
- Normalize text and address duplicate records, conflicting labels and train-test overlap.
- Document data quality checks and preprocessing decisions.

### 4.2 Article Classification
- Develop a TF-IDF and Logistic Regression baseline classifier.
- Generate sentence embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- Train and evaluate an embedding-based classifier.
- Compare traditional lexical features with semantic representations.

### 4.3 Model Evaluation and Experimentation
- Evaluate models using accuracy, precision, recall, macro F1 and weighted F1.
- Conduct confusion matrix analysis and qualitative error inspection.
- Investigate disagreements between classification models.
- Experiment with probability-based soft voting to combine model predictions.
- Evaluate the selected ensemble on the held-out test dataset.

### 4.4 Contextual Advertising and Brand Suitability
- Create a simulated advertising campaign dataset.
- Generate semantic representations of campaign descriptions and target keywords.
- Calculate cosine similarity between article and campaign embeddings.
- Rank campaigns according to semantic relevance.
- Apply campaign eligibility rules, including category restrictions, excluded keywords and campaign status.
- Introduce a review decision for articles that fall below campaign-specific relevance thresholds.

### 4.5 Planned Data Engineering and Application Development
- Develop a structured data pipeline for article and campaign processing.
- Introduce PostgreSQL for storing articles, campaign configurations and matching results.
- Explore dbt for data transformation and analytical modelling.
- Use Apache Airflow for workflow orchestration.
- Develop an interactive application for exploring classification results, campaign matches and brand suitability decisions.



## 5. Scope

The initial project will use appropriately licensed public text datasets and a documented simulated advertising campaign dataset.

The project will focus on content-based contextual analysis. It will not collect personal browsing histories, behavioural profiles or personally identifiable information.

## 6. Success Criteria

The project will be considered successful when it can:

- Reproducibly process the selected article dataset.
- Classify articles and report classification performance.
- Compare traditional text representations with semantic embeddings.
- Rank advertising campaigns against article content.
- Apply and evaluate explicit suitability rules.
- Explain its recommendations and limitations.
- Provide an interactive demonstration.

## 7. Limitations

The project is an independent simulation and does not use any companies proprietary models, data or systems.

Its results will depend on the selected datasets, campaign descriptions, relevance labels and evaluation methodology. Similarity scores will not automatically be interpreted as probabilities of advertising suitability.

## 8. Intended Audience

The project is designed to demonstrate practical capabilities in data science, NLP, machine learning, experimentation, data engineering and analytical communication.
