from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from rag.document_loader import load_web_documents, load_pdf_document
from rag.db_connection import get_vector_store
from rag.indexing import split_and_index
from rag.retrieval import get_retrieval_tool
from rag.generator import create_rag_agent

def main():
    # 1. init ollama embeddings and model
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    model = ChatOllama(model="llama3.1")

    # 2. Connect to Database
    vector_store = get_vector_store(embedding_function=embeddings)

    # 3. Load Documents (Choose either Web or PDF)
    print("Loading documents...")
    
    # Option A: Load from Web
    # url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    # docs = load_web_documents(url)
    
    # Option B: Load from PDF (assuming you have a file named 'example.pdf' in root)
    pdf_path = "fake_company.pdf" 
    docs = load_pdf_document(pdf_path)

    # 4. Index the data into the vector store
    print("Splitting and indexing data...")
    split_and_index(vector_store, docs)

    # 5. Setup the retrieval tool connected to your vector store
    retrieve_tool = get_retrieval_tool(vector_store)
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
            response = agent.invoke({"messages": [("user", query)]})
            
            # Print the response content
            if "messages" in response:
                print(f"Assistant: {response['messages'][-1].content}")
            else:
                print(f"Assistant: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
