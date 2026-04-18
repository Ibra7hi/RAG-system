import chromadb
from langchain_community.vectorstores import Chroma

def get_vector_store(embedding_function, host="localhost", port=8000, collection_name="my_rag_collection"):
    """Connects to ChromaDB and returns a Langchain vector store."""
    chroma_client = chromadb.HttpClient(host=host, port=port)
    
    vector_store = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    
    return vector_store
