"""
MCP Server for the RAG system.

Exposes the RAG retrieval pipeline as a discoverable MCP tool.
Any MCP-compatible client (LangChain agent, Claude Desktop, etc.)
can connect and use these tools dynamically — no hardcoding needed.

Run:  python mcp_server.py
"""

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from langchain_ollama import OllamaEmbeddings

from rag.db_connection import get_vector_store
from rag.hybrid_retriever import create_hybrid_retriever

# ── Initialize the MCP Server ──────────────────────────────────────
mcp = FastMCP("RAG-Tools-Server")

# ── Initialize the RAG backend (runs once at startup) ──────────────
print("🔧 MCP Server: Initializing RAG backend...")

embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
vector_store = get_vector_store(embedding_function=embeddings)
hybrid_retriever = create_hybrid_retriever(vector_store)

print("✅ MCP Server: RAG backend ready.\n")


# ── MCP Tools ──────────────────────────────────────────────────────

@mcp.tool()
def retrieve_context(query: str, metadata_filter: Optional[str] = None) -> str:
    """Retrieve information from indexed documents to help answer a query.

    Uses a hybrid search (BM25 keyword + semantic vector) for best results.

    Args:
        query: The search query to find relevant context.
        metadata_filter: Optional JSON string of metadata to filter by,
            e.g. '{"company": "FakeCorp", "doc_type": "handbook"}'.
            Use this when the user asks about a specific company,
            document, or category.

    Returns:
        A formatted string of the most relevant document chunks,
        or a message saying no matches were found.
    """
    # Parse the optional metadata filter from JSON string
    filter_dict = None
    if metadata_filter:
        try:
            filter_dict = json.loads(metadata_filter)
        except json.JSONDecodeError:
            return f"Error: metadata_filter is not valid JSON: {metadata_filter}"

    # Run the hybrid retriever (BM25 + semantic with RRF fusion)
    retrieved_docs = hybrid_retriever.invoke(query, metadata_filter=filter_dict)

    if not retrieved_docs:
        return "No matching documents found after filtering."

    # Serialize results for the agent to read
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized


# ── Run the server ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting MCP Server on http://0.0.0.0:8081/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8081)
