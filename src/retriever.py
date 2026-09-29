import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNKS_PATH = "data/processed/chunks.jsonl"
INDEX_PATH = "data/processed/constitution.faiss"


# ---------------------------------------------------------
# Constitution-specific query expansion
# ---------------------------------------------------------

ARTICLE_TERMS = {
    "14": "equality before law",
    "15": "prohibition of discrimination",
    "16": "equality of opportunity in matters of public employment",
    "17": "abolition of untouchability",
    "18": "abolition of titles",
    "19": "protection of certain rights regarding freedom of speech",
    "20": "protection in respect of conviction for offences",
    "21": "protection of life and personal liberty",
    "21a": "right to education",
    "22": "protection against arrest and detention",
    "23": "traffic in human beings and forced labour",
    "24": "employment of children in factories",
    "25": "freedom of conscience and free profession practice propagation of religion",
    "26": "freedom to manage religious affairs",
    "27": "freedom as to payment of taxes for promotion of religion",
    "28": "freedom as to attendance at religious instruction",
    "29": "protection of interests of minorities",
    "30": "right of minorities to establish and administer educational institutions",
    "32": "remedies for enforcement of fundamental rights",
}


def load_chunks(path: str) -> list[dict]:
    """Load document chunks from JSONL."""

    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


def expand_query(query: str) -> str:
    """
    Expand queries containing a known constitutional article number.

    Example:
        "What does Article 21 say?"
    becomes:
        "What does Article 21 say?
         Article 21 protection of life and personal liberty"
    """

    article_match = re.search(
        r"\barticle\s+(\d+[a-z]?)\b",
        query.lower(),
    )

    if not article_match:
        return query

    article_number = article_match.group(1)

    description = ARTICLE_TERMS.get(article_number)

    if not description:
        return query

    expanded_query = (
        f"{query} "
        f"Article {article_number} "
        f"{description}"
    )

    return expanded_query


class HybridRetriever:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        chunks_path: str = CHUNKS_PATH,
        index_path: str = INDEX_PATH,
    ):
        print("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

        print("Loading chunks...")

        self.chunks = load_chunks(chunks_path)

        print("Loading FAISS index...")

        self.index = faiss.read_index(index_path)

        print("Building BM25 index...")

        tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

        print(
            f"Hybrid retriever ready: "
            f"{len(self.chunks)} chunks"
        )

    # ---------------------------------------------------------
    # Semantic / FAISS search
    # ---------------------------------------------------------

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[int]:

        expanded_query = expand_query(query)

        query_embedding = self.model.encode(
            [expanded_query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        _, indices = self.index.search(
            query_embedding,
            top_k,
        )

        return [
            int(index)
            for index in indices[0]
            if index >= 0
        ]

    # ---------------------------------------------------------
    # BM25 keyword search
    # ---------------------------------------------------------

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[int]:

        expanded_query = expand_query(query)

        query_tokens = tokenize(expanded_query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            int(index)
            for index in ranked_indices
        ]

    # ---------------------------------------------------------
    # Reciprocal Rank Fusion
    # ---------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: list[list[int]],
        k: int = 60,
    ) -> list[tuple[int, float]]:

        scores = {}

        for result_list in result_lists:

            for rank, doc_index in enumerate(
                result_list,
                start=1,
            ):

                scores[doc_index] = (
                    scores.get(doc_index, 0.0)
                    + 1.0 / (k + rank)
                )

        ranked_results = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked_results

    # ---------------------------------------------------------
    # Hybrid retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        # Retrieve more candidates than we finally return.
        semantic_results = self.semantic_search(
            query,
            top_k=10,
        )

        keyword_results = self.keyword_search(
            query,
            top_k=10,
        )

        fused_results = self.reciprocal_rank_fusion(
            [
                semantic_results,
                keyword_results,
            ]
        )

        results = []

        for doc_index, rrf_score in fused_results[:top_k]:

            chunk = self.chunks[doc_index].copy()

            chunk["rrf_score"] = rrf_score

            results.append(chunk)

        return results


# ---------------------------------------------------------
# Manual testing
# ---------------------------------------------------------

if __name__ == "__main__":

    retriever = HybridRetriever()

    queries = [
        "What are the Fundamental Rights guaranteed by the Constitution?",
        "What does Article 21 say about protection of life and personal liberty?",
        "How is the President of India elected?",
        "What does Article 14 say?",
        "What is Article 32?",
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        expanded = expand_query(query)

        if expanded != query:
            print("\nExpanded Query:")
            print(expanded)

        results = retriever.retrieve(
            query,
            top_k=3,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n--- Hybrid Result {rank} ---"
            )

            print(
                f"RRF Score: "
                f"{result['rrf_score']:.6f}"
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