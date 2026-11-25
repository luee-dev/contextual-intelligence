
# Campaign Data Dictionary

## 1. Purpose

This document defines the structure of the simulated advertising campaign dataset used in the Contextual Intelligence project.

The dataset represents hypothetical advertisers, their campaign objectives, contextual preferences and brand suitability restrictions.

It is synthetic and created exclusively for educational and portfolio demonstration purposes.

## 2. Dataset Structure

| Field | Data Type | Description |
|---|---|---|
| campaign_id | string | Unique campaign identifier |
| advertiser | string | Hypothetical advertiser name |
| campaign_name | string | Name of the advertising campaign |
| campaign_description | string | Description of the product, service or campaign objective |
| target_categories | list | News categories suitable for the campaign |
| target_keywords | list | Keywords indicating relevant content |
| excluded_keywords | list | Keywords indicating content that should be excluded |
| excluded_categories | list | News categories excluded from placement |
| minimum_relevance_score | float | Minimum semantic similarity score required for a match |
| active | boolean | Indicates whether the campaign is active |

## 3. News Categories

The campaign dataset uses the four categories supported by the classification model:

- World
- Sports
- Business
- Sci/Tech

## 4. Suitability Logic

A campaign should only be considered for placement when:

1. It is active.
2. The article does not belong to an excluded category.
3. The article does not contain an excluded keyword.
4. Its contextual relevance score meets the campaign threshold.

Campaigns that fail suitability checks should not be recommended for placement.

## 5. Important Limitations

- Campaigns and advertisers are fictional.
- Keywords are simplified indicators, not a complete brand-safety taxonomy.
- Keyword matching can miss context, negation and ambiguous meanings.
- Category classification alone cannot determine whether content is suitable for a particular advertiser.
- Semantic similarity does not guarantee advertising relevance.

The system is a prototype and should not be interpreted as a production-ready brand-safety solution.
