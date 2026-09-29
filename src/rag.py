from llm import OpenRouterProvider
from retriever import HybridRetriever


SYSTEM_PROMPT = """
You are Constitution AI, a factual assistant for the Constitution of India.

Your answers must be grounded ONLY in the Constitution context provided
by the application.

Rules:

1. Use only the supplied context to answer the question.
2. Do not invent constitutional provisions, articles, clauses, amendments,
   dates, or facts.
3. If the supplied context does not contain enough information to answer
   the question, clearly say that the available Constitution context is
   insufficient.
4. Do not use your general knowledge to fill missing information.
5. Keep the answer clear and concise.
6. Cite the relevant Constitution page numbers provided in the context.
"""


class ConstitutionRAG:

    def __init__(self):

        print("Initializing retriever...")

        self.retriever = HybridRetriever()

        print("Initializing LLM...")

        self.llm = OpenRouterProvider()

    def build_context(
        self,
        results: list[dict],
    ) -> str:

        context_parts = []

        for result in results:

            context_parts.append(
                f"""
SOURCE
Document: {result['source']}
PDF Page: {result['page']}
Chunk ID: {result['chunk_id']}

CONTENT:
{result['text']}
"""
            )

        return "\n".join(context_parts)

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:

        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        context = self.build_context(results)

        prompt = f"""
Answer the following question using ONLY the Constitution context below.

QUESTION:
{question}

CONSTITUTION CONTEXT:
{context}

Remember:
- Use only the supplied context.
- If the context is insufficient, say so.
- Include citations such as [Constitution 2024, PDF page X].
"""

        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        sources = [
            {
                "page": result["page"],
                "chunk_id": result["chunk_id"],
                "score": result["rrf_score"],
            }
            for result in results
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":

    rag = ConstitutionRAG()

    questions = [
        "What does Article 21 say about protection of life and personal liberty?",
        "What are the Fundamental Rights mentioned in Part III?",
    ]

    for question in questions:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        response = rag.answer(
            question,
            top_k=5,
        )

        print("\nANSWER:\n")
        print(response["answer"])

        print("\nRETRIEVED SOURCES:")

        for source in response["sources"]:

            print(
                f"- Page {source['page']} "
                f"| {source['chunk_id']} "
                f"| RRF={source['score']:.6f}"
            )