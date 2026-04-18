
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from rag.indexing import load_and_index
from rag.retrivel import get_retrieval_tool
from rag.generator import create_rag_agent

# Initialize Ollama Models
embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = init_chat_model(
    "auto",
    model_provider="openrouter",
)
vector_store = Chroma(embedding_function=embeddings)

# 1. Index the data into the vector store
print("Indexing data...")
load_and_index(vector_store)

# 2. Setup the retrieval tool connected to your vector store
retrieve_tool = get_retrieval_tool(vector_store)
tools = [retrieve_tool]

# 3. Create the Agent
agent = create_rag_agent(model, tools)

print("Agent is ready!")
