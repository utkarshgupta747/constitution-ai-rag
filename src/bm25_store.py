import json
import re

from rank_bm25 import BM25Okapi


CHUNKS_PATH = "data/processed/chunks.jsonl"


def load_chunks(path: str) -> list[dict]:
    """Load chunks from JSONL."""

    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


class BM25Retriever:

    def __init__(
        self,
        chunks_path: str = CHUNKS_PATH,
    ):
        print("Loading chunks for BM25...")

        self.chunks = load_chunks(chunks_path)

        self.tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

        print(
            f"BM25 ready: {len(self.chunks)} documents"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve documents using BM25 keyword matching."""

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            chunk = self.chunks[index].copy()

            chunk["score"] = float(
                scores[index]
            )

            results.append(chunk)

        return results


if __name__ == "__main__":

    retriever = BM25Retriever()

    queries = [
        "What are the Fundamental Rights guaranteed by the Constitution?",
        "What does Article 21 say about protection of life and personal liberty?",
        "How is the President of India elected?",
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = retriever.retrieve(
            query,
            top_k=3,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n--- Result {rank} ---"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Chunk ID: {result['chunk_id']}"
            )

            print(
                result["text"][:800]
            )