# RAG Intelligence System: Study Guide & Documentation

This document serves as a comprehensive overview of the RAG Intelligence Enterprise AI Assistant. It is designed to be fed into an AI model so the model can test your understanding of the system's architecture, data flow, and advanced components.

---

## Section 1: High-Level Architecture
The system is a production-ready Retrieval-Augmented Generation (RAG) assistant built using a modern AI stack. It separates the "brain" (Orchestrator) from the "tools" (MCP Server).

*   **Frontend**: Built with Next.js (React), running on `localhost:3000`. It features a dynamic chat interface with tailored caching configurations.
*   **API Gateway & Orchestrator**: Built with FastAPI (`app.py`), running on `localhost:8080`.
*   **Agent Framework**: LangGraph ReAct Agent. It acts as the decision-maker, maintaining conversation state and calling tools.
*   **Tool Provider (MCP Server)**: `mcp_server.py`. It exposes the heavy RAG logic (embeddings, retrievers, DB connections) to the Orchestrator dynamically via the Model Context Protocol (MCP).
*   **Vector Database**: PostgreSQL with the `pgvector` extension (running via Docker), handling both vector storage and metadata filtering using JSONB.
*   **Models**: 
    *   *LLM Brain*: OpenRouter (e.g., cloud models) or local Ollama.
    *   *Embeddings*: Nomic Embed Text (`nomic-embed-text-v2-moe`) running locally via Ollama.

---

## Section 2: Core Components

### 2.1 The Orchestrator (`app.py`)
This is a clean-architecture FastAPI backend. It explicitly does **not** know about the database, document chunking, or embeddings.
*   **Responsibilities**: 
    *   Serves the HTTP API (`/api/chat`).
    *   Connects to the OpenRouter/OpenAI API.
    *   Spawns the MCP server as a managed subprocess (`stdio` transport) and dynamically discovers tools (e.g., `retrieve_context`).
    *   Initializes the LangGraph ReAct agent with persistent conversational memory (checkpointing).

### 2.2 The MCP Server (`mcp_server.py`)
This script owns the heavy RAG dependencies. It uses `FastMCP` to expose tools to the LangChain agent.
*   **Responsibilities**:
    *   Initializes the `OllamaEmbeddings` and connects to the PGVector database.
    *   Exposes the `@mcp.tool()` named `retrieve_context`.
    *   Applies Query Rewriting before passing the search term to the retriever.

### 2.3 The Hybrid Retriever (`rag/hybrid_retriever.py`)
Provides advanced search capabilities by combining two strategies:
*   **BM25 (Keyword Search)**: Loads documents into memory from PostgreSQL to build a sparse keyword index. Good for exact term matches.
*   **Semantic Search (Vector Search)**: Uses PGVector to find conceptually similar chunks.
*   **Pre-filtering**: 
    *   When metadata filters are provided, it filters the documents *before* searching.
    *   It dynamically builds a localized BM25 index exclusively on the filtered subset.
    *   It uses PostgreSQL's native JSONB querying to filter the vector store natively.
*   Both results are combined using an `EnsembleRetriever`.

### 2.4 Query Rewriting (`rag/query_rewriter.py`)
To optimize retrieval quality, raw user queries are first rewritten using an LLM. This handles vague queries, co-reference resolution (e.g., resolving "it" or "they" to previous entities), and extracts optimized search keywords before the retriever does its job.

### 2.5 Re-ranking Layer (`rag/retrieval.py`)
After the hybrid retriever pulls a broad set of candidate documents, an ultra-lightweight Cross-Encoder Re-ranker (`Flashrank / TinyBERT`) evaluates each candidate against the query. The re-ranker acts as a highly accurate post-retrieval filter, scoring the documents and keeping only the top `n` matches before passing the context to the LLM.

---

## Section 3: Data Flow (Step-by-Step)

When a user types a message in the Next.js frontend, the following sequence occurs:

1.  **Request**: Next.js sends an HTTP POST request to FastAPI (`/api/chat`) containing the user's query.
2.  **Agent Invocation**: `app.py` passes the query to the LangGraph ReAct Agent. The agent checks its conversation memory.
3.  **Tool Decision**: The ReAct Agent analyzes the query and decides it needs external information. It decides to call the dynamically discovered `retrieve_context` tool.
4.  **MCP Execution**: The request is routed to the `mcp_server.py` subprocess.
5.  **Query Rewriting**: Inside the MCP server, the raw query is rewritten by the `query_rewriter` to improve search accuracy.
6.  **Hybrid Retrieval**: The rewritten query is sent to the `DynamicHybridRetriever`.
    *   *If metadata filters exist*, pre-filtering occurs on both BM25 and PGVector.
    *   BM25 and Semantic search results are retrieved and fused to form a candidate pool.
7.  **Re-ranking**: The candidate pool is passed through an ultra-lightweight Cross-Encoder model (`Flashrank / TinyBERT`). The re-ranker scores the relevance of each document and compresses the list down to the absolute best matches.
8.  **Context Return**: The MCP Server formats the retrieved, re-ranked documents as a string and returns them to the Orchestrator.
9.  **Generation**: The ReAct Agent reads the injected context, decides it has enough information, and generates a final, human-readable response.
10. **Response**: FastAPI returns the text to the Next.js frontend to be displayed.

---

## Section 4: AI Testing Instructions

*Instructions for the AI evaluating the student:*

Please test the student's understanding of this system by asking targeted questions based on the documentation above. Good areas to test include:
1.  **Architecture**: Ask why the Orchestrator (`app.py`) and the RAG logic (`mcp_server.py`) are strictly separated, and how they communicate.
2.  **Retrieval Strategy**: Ask them to explain the difference between BM25 and Semantic search, and why a Hybrid approach is used.
3.  **Pre-filtering**: Ask how metadata filtering works in the Hybrid Retriever and why it is done *before* the search (Pre-filtering) rather than after.
4.  **Query Lifecycle**: Ask them to trace a user's question from the Next.js frontend all the way through the LangGraph Agent, Query Rewriting, and Database, back to the user.
5.  **State Management**: Ask how the agent remembers previous parts of the conversation.
6.  **Re-ranking**: Ask why a Cross-Encoder Re-ranker is used *after* the initial hybrid retrieval, and how it improves the context quality given to the LLM.
