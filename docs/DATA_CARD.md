# Data Card: AG News

## 1. Dataset Identification

**Dataset name:** AG News Topic Classification Dataset

**Source:** Hugging Face Datasets

**Dataset identifier:** `fancyzhx/ag_news`

**Original dataset:** AG's News Corpus

**Language:** English

**Task:** Multiclass text classification

## 2. Dataset Description

AG News is a news topic classification dataset constructed from news articles gathered by ComeToMyHead, an academic news search engine.

The dataset contains four topic categories:

- World
- Sports
- Business
- Sci/Tech

The examples contain a news title and a short description, together with a category label.

## 3. Dataset Size

| Split | Records |
|---|---:|
| Training | 120,000 |
| Testing | 7,600 |
| Total | 127,600 |

## 4. Intended Project Uses

This dataset will initially support:

- Exploratory data analysis
- Text preprocessing
- Feature extraction
- Supervised topic classification
- Model comparison
- Classification evaluation

It will not be treated as a complete representation of the article content available on a publisher's website.

## 5. Limitations

- The dataset contains titles and descriptions rather than full article bodies.
- It is English-language only.
- Its four categories do not cover every contextual advertising topic.
- Its examples reflect the original collection period and sources.
- Topic classification performance does not establish advertising relevance or brand suitability.

## 6. Licensing and Usage Considerations

The original AG News dataset describes its use as intended for research and non-commercial activities.

This project uses the dataset for educational and portfolio experimentation. The dataset records will not be redistributed through this repository.

Users should consult the original dataset terms before using the data for other purposes.

## 7. Citation

Xiang Zhang, Junbo Zhao and Yann LeCun (2015).

Character-level Convolutional Networks for Text Classification.

Advances in Neural Information Processing Systems 28.

Original dataset: AG's News Corpus.

Dataset access: https://huggingface.co/datasets/fancyzhx/ag_news
