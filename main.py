from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from rag.document_loader import load_web_documents
from rag.db_connection import get_vector_store
from rag.indexing import split_and_index
from rag.retrieval import get_retrieval_tool
from rag.generator import create_rag_agent

def main():
    # 1. Initialize Ollama Models
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    model = ChatOllama(model="llama3.1")

    # 2. Connect to Database
    vector_store = get_vector_store(embedding_function=embeddings)

    # 3. Load Documents
    print("Loading documents...")
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    docs = load_web_documents(url)

    # 4. Index the data into the vector store
    print("Splitting and indexing data...")
    split_and_index(vector_store, docs)

    # 5. Setup the retrieval tool connected to your vector store
    retrieve_tool = get_retrieval_tool(vector_store)
    tools = [retrieve_tool]

    # 6. Create the Agent
    agent = create_rag_agent(model, tools)

    print("Agent is ready!")

if __name__ == "__main__":
    main()
