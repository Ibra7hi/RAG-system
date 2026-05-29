from langchain_community.vectorstores.pgvector import PGVector

def get_vector_store(embedding_function, host="localhost", port=6024, collection_name="my_rag_collection"):
    connection_string = f"postgresql+psycopg2://myuser:mypassword@{host}:{port}/rag_db"

    vector_store = PGVector(
        connection_string=connection_string,
        embedding_function=embedding_function,
        collection_name=collection_name,
        use_jsonb=True
    )
    
    return vector_store
