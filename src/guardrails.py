import re


# ---------------------------------------------------------
# Constitution AI scope
# ---------------------------------------------------------

SCOPE_KEYWORDS = [
    "constitution",
    "constitutional",
    "article",
    "articles",
    "fundamental right",
    "fundamental rights",
    "fundamental duty",
    "fundamental duties",
    "directive principle",
    "directive principles",
    "preamble",
    "amendment",
    "amendments",
    "schedule",
    "schedules",
    "parliament",
    "president of india",
    "vice president",
    "prime minister",
    "supreme court",
    "high court",
    "judiciary",
    "citizenship",
    "election",
    "emergency",
    "writ",
    "ordinance",
    "bill",
    "legislative",
    "constitutional law",
]


# ---------------------------------------------------------
# Prompt injection / malicious indicators
# ---------------------------------------------------------

MALICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?system\s+instructions",
    r"ignore\s+your\s+instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?your\s+system\s+prompt",
    r"reveal\s+hidden\s+instructions",
    r"bypass\s+(your\s+)?safety",
    r"disable\s+(your\s+)?safety",
    r"jailbreak",
]


# ---------------------------------------------------------
# Clearly unrelated topics
# ---------------------------------------------------------

OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b",
    r"\btemperature\b",
    r"\bcricket\b",
    r"\bfootball\b",
    r"\bsoccer\b",
    r"\brecipe\b",
    r"\bcooking\b",
    r"\bmovie\b",
    r"\bmovies\b",
    r"\bsong\b",
    r"\btravel\b",
    r"\bflight\b",
    r"\bhotel\b",
]


def is_malicious_query(query: str) -> bool:
    """
    Detect obvious prompt injection or malicious requests.
    """

    normalized = query.lower().strip()

    return any(
        re.search(pattern, normalized)
        for pattern in MALICIOUS_PATTERNS
    )


def is_constitution_related(query: str) -> bool:
    """
    Determine whether a query is related to the
    Constitution/legal domain supported by the application.
    """

    normalized = query.lower().strip()

    # Explicit Article number
    if re.search(
        r"\barticle\s+\d+[a-z]?\b",
        normalized,
    ):
        return True

    # Known Constitution/legal terminology
    return any(
        keyword in normalized
        for keyword in SCOPE_KEYWORDS
    )


def is_obviously_out_of_scope(query: str) -> bool:
    """
    Detect obvious unrelated requests.
    """

    normalized = query.lower().strip()

    return any(
        re.search(pattern, normalized)
        for pattern in OUT_OF_SCOPE_PATTERNS
    )


def check_query(query: str) -> str:
    """
    Return one of:

    - safe
    - malicious
    - out_of_scope
    - constitution
    """

    if not query or not query.strip():
        return "out_of_scope"

    if is_malicious_query(query):
        return "malicious"

    if is_obviously_out_of_scope(query):
        return "out_of_scope"

    if is_constitution_related(query):
        return "constitution"

    # Conservative default.
    return "out_of_scope"