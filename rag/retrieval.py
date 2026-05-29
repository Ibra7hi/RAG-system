from langchain_core.tools import tool


from typing import Optional, Dict, Any

def create_retrieval_tool(retriever):
    """
    Creates a retrieval tool that the RAG agent can use to search for relevant documents.

    Args:
        retriever: Any LangChain retriever (hybrid, semantic-only, etc.)
                   Must support .invoke(query) and return List[Document]
    """

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str, metadata_filter: Optional[Dict[str, Any]] = None):
        """Retrieve information to help answer a query.
        
        Args:
            query: The search query to find relevant context.
            metadata_filter: Optional dictionary of metadata to filter by (e.g. {"company": "FakeCorp", "doc_type": "handbook"}). Use this when the user asks about a specific company, document, or category.
        """
        # Run the retriever (hybrid: BM25 + semantic search with RRF fusion)
        # We pass the metadata filter down to the DynamicHybridRetriever for strict pre-filtering!
        retrieved_docs = retriever.invoke(query, metadata_filter=metadata_filter)

        if not retrieved_docs:
            return "No matching documents found after filtering.", []

        # Serialize results for the agent to read
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context
