import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings
from rag.api_models import get_fast_llm

from rag.db_connection import get_vector_store
from rag.retrieval import get_retrieval_tool
from rag.generator import create_rag_agent

app = FastAPI(title="RAG Chat API")

# Add CORS middleware to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 1: Initialize Embeddings and LLM
embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
model = get_fast_llm()

# Step 2: Initialize Vector Store Connection
vector_store = get_vector_store(embedding_function=embeddings)

# Step 3: Initialize Retrieval Tool and Agent
retrieve_tool = get_retrieval_tool(vector_store)
tools = [retrieve_tool]
agent = create_rag_agent(model, tools)

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = agent.invoke({"messages": [("user", request.query)]})
        # Step 4: Parse Agent Response
        if "messages" in response:
             return {"response": response['messages'][-1].content}
        else:
             return {"response": str(response)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting API Server on http://localhost:8080")
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
