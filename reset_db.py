from rag.db_connection import get_vector_store
from langchain_ollama import OllamaEmbeddings

def reset_database():
    # We need the embedding function to initialize the Chroma object
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    
    print("Connecting to ChromaDB...")
    vector_store = get_vector_store(embedding_function=embeddings)
    
    print("Deleting collection 'my_rag_collection'...")
    try:
        vector_store.delete_collection()
        print("Success: Database cleared.")
    except Exception as e:
        print(f"Error: Could not clear database: {e}")

if __name__ == "__main__":
    reset_database()
