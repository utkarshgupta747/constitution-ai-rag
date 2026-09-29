from src.retriever import HybridRetriever


def test_article_21_retrieval():

    retriever = HybridRetriever()

    results = retriever.retrieve(
        "What does Article 21 say?",
        top_k=5
    )

    assert len(results) == 5

    pages = [
        result["page"]
        for result in results
    ]

    assert 42 in pages