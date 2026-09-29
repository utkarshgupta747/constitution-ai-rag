import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(path: str) -> list[dict]:
    """Load chunks from JSONL."""

    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def create_embeddings(
    chunks: list[dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """Generate normalized embeddings for all chunks."""

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    return np.asarray(embeddings, dtype="float32")


def save_embeddings(
    embeddings: np.ndarray,
    output_path: str,
) -> None:
    """Save embeddings to disk."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.save(output, embeddings)


if __name__ == "__main__":
    chunks_path = "data/processed/chunks.jsonl"
    embeddings_path = "data/processed/embeddings.npy"

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunks...")

    chunks = load_chunks(chunks_path)

    print(f"Chunks loaded: {len(chunks)}")

    print("Creating embeddings...")

    embeddings = create_embeddings(chunks, model)

    save_embeddings(embeddings, embeddings_path)

    print("\nEmbedding generation complete.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {embeddings_path}")