import re


CONSTITUTION_KEYWORDS = [
    "article",
    "articles",
    "constitution",
    "fundamental right",
    "fundamental rights",
    "directive principle",
    "directive principles",
    "preamble",
    "amendment",
    "schedule",
    "part iii",
    "part iv",
    "part v",
    "president",
    "vice-president",
    "parliament",
    "supreme court",
    "high court",
    "citizenship",
    "election",
    "emergency",
    "fundamental duties",
]


WEB_INDICATORS = [
    "latest",
    "recent",
    "today",
    "current",
    "news",
    "new judgment",
    "recent judgment",
    "recent ruling",
    "2026",
    "2025",
]


def route_query(query: str) -> str:
    """
    Determine whether a query should use the Constitution
    corpus or web search.
    """

    normalized = query.lower().strip()

    # Current/recent information should go to web search.
    for indicator in WEB_INDICATORS:

        if indicator in normalized:
            return "web"

    # Explicit constitutional terminology goes to RAG.
    for keyword in CONSTITUTION_KEYWORDS:

        if keyword in normalized:
            return "constitution"

    # Article-number pattern such as Article 21.
    if re.search(
        r"\barticle\s+\d+[a-z]?\b",
        normalized,
    ):
        return "constitution"

    # Default to Constitution because this is primarily
    # a Constitution-focused application.
    return "constitution"


if __name__ == "__main__":

    test_queries = [
        "What does Article 21 say?",
        "What are Fundamental Rights?",
        "How is the President elected?",
        "What is the latest Supreme Court judgment on Article 21?",
        "What happened in the news today?",
        "Explain the Preamble",
    ]

    for query in test_queries:

        route = route_query(query)

        print(
            f"{query}\n"
            f"  -> {route}\n"
        )