from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_and_index(vector_store, docs, chunk_size=500, chunk_overlap=75,):
    """Splits documents and indexes them into the vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)
    
    return all_splits