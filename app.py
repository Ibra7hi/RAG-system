import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from rag.db_connection import get_vector_store
from rag.hybrid_retriever import create_hybrid_retriever
from rag.retrieval import create_retrieval_tool
from rag.generator import create_rag_agent

app = FastAPI(title="RAG Chat API")

# Add CORS middleware to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local network testing
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from a local .env file if it exists
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

# Step 1: Initialize Embeddings (local) and LLM (OpenRouter API)
embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")

# We use the special "openrouter/free" model which automatically routes to the best active free model
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("\n⚠️ WARNING: OPENROUTER_API_KEY is not set! The server will start, but requests will fail.")
    # Use a dummy key so the OpenAI client initializes without crashing
    api_key = "dummy-key-set-openrouter-api-key-in-env-file"

# Crucial: Set the OPENAI_API_KEY environment variable so the underlying openai client
# is guaranteed to find it and send the correct Authorization header.
os.environ["OPENAI_API_KEY"] = api_key

model = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=api_key,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Step 2: Initialize Vector Store Connection
vector_store = get_vector_store(embedding_function=embeddings)

# Step 3: Build Hybrid Retriever (BM25 + Semantic) and Create Tool
hybrid_retriever = create_hybrid_retriever(vector_store)
retrieve_tool = create_retrieval_tool(hybrid_retriever)
tools = [retrieve_tool]
agent = create_rag_agent(model, tools)

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": "api_user_session"}}
        response = agent.invoke({"messages": [("user", request.query)]}, config=config)
        # Step 4: Parse Agent Response
        # Walk backwards through messages to find the final AI text response
        # (skip tool call messages and tool result messages)
        if "messages" in response:
            for msg in reversed(response["messages"]):
                if hasattr(msg, 'content') and msg.content and not getattr(msg, 'tool_calls', None):
                    if msg.type == "ai":
                        return {"response": msg.content}
            # Fallback: return last message content
            return {"response": response['messages'][-1].content}
        else:
             return {"response": str(response)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting API Server on http://localhost:8080")
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
