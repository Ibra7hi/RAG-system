import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

# -------------------------------------------------------------
# OPENROUTER API CONFIGURATION
# Use this file if you want fast, cloud-based LLM generation.
# -------------------------------------------------------------

def get_fast_llm(model_name="meta-llama/llama-3.1-8b-instruct"):
    """
    Connects to OpenRouter to use powerful, fast cloud models 
    without needing a heavy local GPU.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "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")
    
    if api_key == "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":
        print("⚠️ WARNING: Please replace 'your_openrouter_api_key_here' with your actual OpenRouter Key!")
        
    return ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        # optional: OpenRouter headers for routing stats
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Local RAG Assistant",
        }
    )

def get_embeddings():
    """
    We still use Ollama for embeddings because embedding models 
    are very small, extremely fast locally, and completely free.
    """
    return OllamaEmbeddings(model="nomic-embed-text-v2-moe")
