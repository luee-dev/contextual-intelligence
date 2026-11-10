# Data Dictionary: AG News

## Dataset: AG News

| Field | Data type | Description |
|---|---|---|
| `text` | String | Combined news title and description |
| `label` | Integer | Numeric category assigned to the news example |
| `label_name` | String | Human-readable category name derived from the label |
| `title` | String | Original news headline, retained when available |
| `description` | String | News description, retained when available |

## Category Mapping

| Label | Category |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## Derived Fields

The project may introduce additional fields during preprocessing and feature engineering.

| Field | Description |
|---|---|
| `article_id` | Unique identifier generated within the project |
| `text_clean` | Normalised text used for modelling |
| `word_count` | Number of tokens in the text |
| `character_count` | Number of characters in the text |

Derived fields will be documented as they are implemented.
