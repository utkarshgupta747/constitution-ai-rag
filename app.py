import os

import streamlit as st

from src.graph import ConstitutionGraph


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Constitution AI",
    page_icon="🇮🇳",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🇮🇳 Constitution AI")

st.markdown(
    """
Ask questions about the Constitution of India using
Retrieval-Augmented Generation (RAG) with source citations.
"""
)

st.caption(
    "Constitution RAG + Hybrid Retrieval + Web Search"
)

st.info(
    "For informational purposes only. "
    "This application does not provide legal advice."
)


# ---------------------------------------------------------
# Load Graph
# ---------------------------------------------------------

@st.cache_resource
def load_graph():

    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            os.environ["OPENROUTER_API_KEY"] = (
                st.secrets["OPENROUTER_API_KEY"]
            )

    except Exception:
        # Local development uses .env
        pass

    return ConstitutionGraph()


# ---------------------------------------------------------
# Initialize Graph
# ---------------------------------------------------------

with st.spinner("Loading Constitution AI..."):
    graph = load_graph()


# ---------------------------------------------------------
# Question Input
# ---------------------------------------------------------

question = st.text_area(
    "Ask your question",
    placeholder="Example: What does Article 21 say?",
    height=100,
)

ask_button = st.button(
    "🔍 Ask",
    type="primary",
)


# ---------------------------------------------------------
# Process Question
# ---------------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Researching..."):

            try:

                result = graph.ask(
                    question.strip()
                )

            except Exception as error:

                st.error(
                    "An error occurred while processing "
                    "your question."
                )

                st.exception(error)

                st.stop()


        # -------------------------------------------------
        # Route
        # -------------------------------------------------

        route = result.get(
            "route",
            "unknown",
        )

        if route == "constitution":

            st.success(
                "📖 Constitution RAG"
            )

        elif route == "web":

            st.success(
                "🌐 Web Search"
            )


        # -------------------------------------------------
        # Answer
        # -------------------------------------------------

        st.subheader("Answer")

        answer = result.get(
            "answer",
            "No answer generated.",
        )

        st.markdown(answer)


        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        if sources:

            with st.expander("🔎 View Sources"):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    # -------------------------------------
                    # Constitution source
                    # -------------------------------------

                    if source.get(
                        "source"
                    ) == "Constitution 2024":

                        page = source.get(
                            "page",
                            "?",
                        )

                        st.markdown(
                            f"""
**{index}. Constitution of India, 2024**

PDF page: **{page}**
"""
                        )


                    # -------------------------------------
                    # Web source
                    # -------------------------------------

                    else:

                        title = source.get(
                            "title",
                            "Web source",
                        )

                        url = source.get(
                            "url",
                            "",
                        )

                        source_type = source.get(
                            "type",
                            "secondary",
                        )

                        st.markdown(
                            f"""
**{index}. {title}**

Source type: **{source_type}**

[Open source ↗]({url})
"""
                        )