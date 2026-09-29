# 🇮🇳 Constitution AI

A Retrieval-Augmented Generation (RAG) application for answering questions about the Constitution of India using authoritative constitutional documents, hybrid retrieval, query routing, and source-grounded LLM responses.

## 🎯 Problem Statement

The goal of this project is to build an AI application that can answer questions about the Constitution of India accurately and traceably.

Instead of relying only on an LLM's internal knowledge, the application retrieves relevant constitutional provisions from the Constitution of India (2024) and uses that retrieved context to generate grounded answers with source references.

The application also supports routing between the constitutional knowledge base and web search for queries requiring recent or current information.

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │   Streamlit UI   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    LangGraph     │
                         │  Query Workflow  │
                         └────────┬─────────┘
                                  │
                         ┌────────▼────────┐
                         │    Guardrail     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Constitution                  Out of Scope /
                 Route                     Malicious Query
                    │                           │
                    ▼                           ▼
             ┌─────────────┐               Rejection
             │ Query Router│
             └──────┬──────┘
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
      Constitution       Web Search
      Knowledge Base
             │
             ▼
      ┌───────────────┐
      │ Hybrid Search │
      │               │
      │ FAISS + BM25  │
      └───────┬───────┘
              │
              ▼
       Retrieved Context
              │
              ▼
       ┌──────────────┐
       │ OpenRouter   │
       │ Hosted LLM   │
       └──────┬───────┘
              │
              ▼
       Grounded Answer
        + Source Pages