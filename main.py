from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from rag.document_loader import load_pdf_document
from rag.db_connection import get_vector_store
from rag.indexing import split_and_index
from rag.hybrid_retriever import create_hybrid_retriever
from rag.retrieval import create_retrieval_tool
from rag.generator import create_rag_agent


def main():
    # 1. init ollama embeddings and model
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    model = ChatOllama(model="llama3.1")

    # 2. Connect to Database
    vector_store = get_vector_store(embedding_function=embeddings)

    # 3. Load Documents from PDF
    print("Loading documents...")
    
    # Load from PDF with some custom metadata attached
    pdf_path = "fake_company.pdf" 
    docs = load_pdf_document(pdf_path, custom_metadata={"company": "FakeCorp", "doc_type": "handbook"})

    # 4. Index the data into the vector store
    print("Splitting and indexing data...")
    split_and_index(vector_store, docs, embeddings=embeddings)

    # 5. Build Hybrid Retriever (BM25 + Semantic) and Setup Tool
    hybrid_retriever = create_hybrid_retriever(vector_store)
    retrieve_tool = create_retrieval_tool(hybrid_retriever)
    
    tools = [retrieve_tool]

    # 6. Create the Agent
    agent = create_rag_agent(model, tools)

    print("\n--- RAG Agent is ready! ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        try:
            # Depending on the agent type, the input key might differ. 
            # For most common agents, it's "messages" or "input".
            config = {"configurable": {"thread_id": "cli_user_session"}}
            response = agent.invoke({"messages": [("user", query)]}, config=config)
            
            # Print the response content
            if "messages" in response:
                print(f"Assistant: {response['messages'][-1].content}")
            else:
                print(f"Assistant: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()