
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT_DIR / "data" / "processed" / "ag_news_test.csv"
EMBEDDING_DIR = ROOT_DIR / "models" / "embeddings"

EMBEDDING_PATH = EMBEDDING_DIR / "ag_news_test_embeddings.npy"
METADATA_PATH = EMBEDDING_DIR / "test_embedding_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64
EMBEDDING_DIMENSION = 384


def main():
    """Generate embeddings for the untouched AG News test dataset."""

    print("\n--- LOADING TEST DATASET ---")

    df = pd.read_csv(DATA_PATH)

    texts = df["text"].astype(str).tolist()

    print(f"Records: {len(texts):,}")

    print("\n--- LOADING SENTENCE TRANSFORMER ---")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print(f"Model: {MODEL_NAME}")

    print("\n--- GENERATING TEST EMBEDDINGS ---")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    expected_shape = (len(texts), EMBEDDING_DIMENSION)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    print("\n--- EMBEDDING VALIDATION ---")
    print(f"Shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")
    print(f"Finite values: {np.isfinite(embeddings).all()}")
    print(f"Mean vector norm: {np.linalg.norm(embeddings, axis=1).mean():.6f}")

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDING_PATH, embeddings)

    metadata = {
        "model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "records": int(len(embeddings)),
        "batch_size": BATCH_SIZE,
        "normalized": True,
        "data_source": "data/processed/ag_news_test.csv",
        "embedding_dtype": str(embeddings.dtype),
        "device": "cpu",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n--- SAVING COMPLETED ---")
    print(f"Embeddings: {EMBEDDING_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
