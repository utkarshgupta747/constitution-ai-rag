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
    "liberty",
    "personal liberty",
    "life and personal liberty",
    "deprived",
    "deprive",
    "deprivation",
    "equality before the law",
    "equal protection",
    "freedom of speech",
    "freedom of religion",
    "religious freedom",
    "right to life",
    "right to education",
    "arrest",
    "detention",
    "forced labour",
    "untouchability",
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


# ---------------------------------------------------------
# Natural-language constitutional rights questions
# ---------------------------------------------------------

CONSTITUTIONAL_RIGHT_PATTERNS = [
    "take away someone's liberty",
    "take away someones liberty",
    "take away liberty",
    "deprive someone of liberty",
    "deprive someone of their liberty",
    "deprive a person of liberty",
    "can the government take away",
    "can government take away",
    "government deprive",
    "government deny",
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

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

    # Natural-language constitutional rights questions
    if any(
        pattern in normalized
        for pattern in CONSTITUTIONAL_RIGHT_PATTERNS
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


# ---------------------------------------------------------
# Main guardrail classifier
# ---------------------------------------------------------

def check_query(query: str) -> str:
    """
    Classify the user query into one of:

    - malicious
    - constitution
    - out_of_scope

    Malicious queries are checked first so that prompt
    injection attempts cannot be classified as normal
    Constitution questions.
    """

    normalized = query.lower().strip()

    # 1. Malicious query
    # MALICIOUS_PATTERNS are regex patterns, so use
    # is_malicious_query() instead of plain substring matching.
    if is_malicious_query(normalized):
        return "malicious"

    # 2. Natural-language constitutional rights questions
    for pattern in CONSTITUTIONAL_RIGHT_PATTERNS:
        if pattern in normalized:
            return "constitution"

    # 3. Explicit constitutional keywords
    if is_constitution_related(normalized):
        return "constitution"

    # 4. Explicit out-of-scope patterns
    if is_obviously_out_of_scope(normalized):
        return "out_of_scope"

    # 5. Default
    return "out_of_scope"