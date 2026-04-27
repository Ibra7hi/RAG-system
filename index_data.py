from langchain_ollama import OllamaEmbeddings
from rag.document_loader import load_pdf_document
from rag.db_connection import get_vector_store
from rag.indexing import split_and_index

def index_my_data():
    print("Step 1: Initialize Database Connection")
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    vector_store = get_vector_store(embedding_function=embeddings)

    print("Step 2: Load Target Documents")
    pdf_path = "fake_company.pdf" 
    try:
        docs = load_pdf_document(pdf_path)
        print(f"   -> Successfully loaded {len(docs)} pages.")
    except Exception as e:
        print(f"   -> Error loading PDF: {e}")
        return

    print("Step 3: Chunk and Index Data")
    split_and_index(vector_store, docs)
    
    print("\n✅ Indexing Complete.")

if __name__ == "__main__":
    index_my_data()
