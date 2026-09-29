import json
from pathlib import Path

import faiss
import numpy as np


def load_chunks(path: str) -> list[dict]:
    """Load chunks from JSONL."""

    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index using inner product similarity.

    Embeddings are normalized, so inner product is equivalent
    to cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def save_index(index: faiss.Index, path: str) -> None:
    """Save FAISS index to disk."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output))


def search(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index."""

    scores, indices = index.search(query_embedding, top_k)

    return scores, indices


if __name__ == "__main__":
    embeddings_path = "data/processed/embeddings.npy"
    chunks_path = "data/processed/chunks.jsonl"
    index_path = "data/processed/constitution.faiss"

    embeddings = np.load(embeddings_path)

    chunks = load_chunks(chunks_path)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Chunks: {len(chunks)}")

    index = build_faiss_index(embeddings)

    save_index(index, index_path)

    print(f"FAISS index created.")
    print(f"Total vectors: {index.ntotal}")
    print(f"Index dimension: {index.d}")
    print(f"Saved to: {index_path}")