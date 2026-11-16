
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from contextual_intelligence.classification.train import (
    load_training_data,
)

ROOT_DIR = Path(__file__).resolve().parents[3]

EMBEDDING_DIR = ROOT_DIR / "models" / "embeddings"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_PATH = EMBEDDING_DIR / "ag_news_embeddings.npy"
LABELS_PATH = EMBEDDING_DIR / "ag_news_labels.npy"
METADATA_PATH = EMBEDDING_DIR / "embedding_metadata.json"

BATCH_SIZE = 64
EMBEDDING_DIMENSION = 384


def generate_embeddings(texts):
    """Generate normalized sentence embeddings."""

    print("\n--- LOADING SENTENCE TRANSFORMER ---")

    # model = SentenceTransformer(MODEL_NAME)
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print(f"Model: {MODEL_NAME}")
    print("\n--- GENERATING EMBEDDINGS ---")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (
        len(texts),
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    return embeddings


def save_embeddings(embeddings, labels):
    """Save embeddings, labels and generation metadata."""

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDING_PATH, embeddings)
    np.save(LABELS_PATH, labels)

    metadata = {
        "model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "records": int(len(embeddings)),
        "batch_size": BATCH_SIZE,
        "normalized": True,
        "data_source": "data/processed/ag_news_train_clean.csv",
        "embedding_dtype": str(embeddings.dtype),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n--- SAVING COMPLETED ---")
    print(f"Embeddings: {EMBEDDING_PATH}")
    print(f"Labels: {LABELS_PATH}")
    print(f"Metadata: {METADATA_PATH}")


def main():
    """Generate embeddings for the cleaned AG News dataset."""

    print("\n--- LOADING CLEANED DATASET ---")

    df = load_training_data()

    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy(dtype=np.int64)

    print(f"Records: {len(texts):,}")

    embeddings = generate_embeddings(texts)

    print("\n--- EMBEDDING INFORMATION ---")
    print(f"Shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    save_embeddings(embeddings, labels)

    print("\n--- EMBEDDING GENERATION COMPLETED ---")


if __name__ == "__main__":
    main()
