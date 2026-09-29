from src.router import route_query


def test_constitution_route():
    assert route_query(
        "What does Article 21 say?"
    ) == "constitution"


def test_web_route():
    assert route_query(
        "What is the latest Supreme Court judgment on Article 21?"
    ) == "web"


def test_fundamental_rights_route():
    assert route_query(
        "Explain Fundamental Rights."
    ) == "constitution"