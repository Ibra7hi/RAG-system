from langchain_core.tools import tool

def get_retrieval_tool(vector_store):
    # Step 1: Define Retrieval Tool
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        # Step 2: Execute Similarity Search
        retrieved_docs = vector_store.similarity_search(query, k=10)
        
        # Step 3: Serialize Results
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    return retrieve_context
