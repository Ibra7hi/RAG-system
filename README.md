# RAG Intelligence: Enterprise AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with **LangChain**, **LangGraph**, **MCP (Model Context Protocol)**, **FastAPI**, **PGVector**, and **Next.js**.

Tools are **not hardcoded** — they are exposed via an **MCP Server** and discovered dynamically at runtime by the agent.

---

## 🏗 Architecture

```
┌──────────────────┐     MCP Protocol      ┌──────────────────┐
│                  │   (streamable-http)    │                  │
│  FastAPI App     │◄─────────────────────►│  MCP Server      │
│  (app.py)        │   tools discovered    │  (mcp_server.py) │
│  + MCP Client    │   dynamically         │                  │
│                  │                       │  Exposes:         │
│  LangGraph Agent │                       │  - retrieve_ctx   │
│  uses MCP tools  │                       │  - (future tools) │
└──────────────────┘                       └──────────────────┘
   Port 8080                                  Port 8081
   (HTTP API)                              (MCP endpoint)
```

## 🧩 Tech Stack

*   **LLM Providers:** [Ollama](https://ollama.com/) (Local) OR [OpenRouter](https://openrouter.ai/) (Cloud API)
*   **Embeddings:** Nomic Embed Text (Local via Ollama)
*   **AI Logic:** [LangChain](https://www.langchain.com/) & LangGraph (ReAct Agent)
*   **Tool Protocol:** [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) — tools are discovered, not hardcoded
*   **Vector Database:** [PGVector](https://github.com/pgvector/pgvector) (PostgreSQL + vector search via Docker)
*   **Backend API:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **Frontend UI:** [Next.js](https://nextjs.org/) (React, Tailwind CSS, Framer Motion)

---

## 📂 Project Structure

```text
RAG/
├── docker-compose.yml       # Sets up the PGVector container
├── requirements.txt         # Python dependencies
├── mcp_server.py            # ⭐ MCP Server — exposes RAG tools over MCP protocol
├── app.py                   # FastAPI Server — MCP client that discovers tools dynamically
├── main.py                  # CLI interface — also uses MCP for tool discovery
├── index_data.py            # Script to load, chunk & index PDFs into PGVector
├── reset_db.py              # Script to wipe the database clean
│
├── rag/                     # Core RAG Logic
│   ├── document_loader.py   # Extracts text from PDFs
│   ├── indexing.py          # Chunks text & stores vectors
│   ├── hybrid_retriever.py  # BM25 + Semantic hybrid search with pre-filtering
│   ├── retrieval.py         # LangChain tool wrapper (used inside MCP server)
│   ├── generator.py         # Agent setup & PostgreSQL checkpointer
│   ├── db_connection.py     # Connects to PGVector
│   └── api_models.py        # Cloud Model Configuration (OpenRouter)
│
└── frontend/                # Next.js UI Application
    ├── src/app/page.tsx     # Main chat interface
    └── src/app/globals.css  # UI Styling
```

---

## 🛠 Prerequisites

Before running the project, ensure you have the following installed:
1.  **Python 3.10+** (with a virtual environment set up)
2.  **Node.js 18+** (for the Next.js frontend)
3.  **Docker** (to run PGVector)
4.  **Ollama** (installed locally for embeddings & optional local generation).

**Pull the required Ollama models:**
```bash
ollama pull llama3.1
ollama pull nomic-embed-text-v2-moe
```

*(Note: If you plan to use OpenRouter for generation, add your `OPENROUTER_API_KEY` to `.env`)*

---

## 🚀 Execution Steps

You will need **four** terminal windows to run the different parts of the stack.

### Step 1: Initialize the Database
```bash
docker compose up -d
```

### Step 2: Index Your Data (first time only)
```bash
python3 index_data.py
```

### Step 3: Start the MCP Server
```bash
python3 mcp_server.py
```
This starts the MCP tool server on **http://localhost:8081/mcp**.

### Step 4: Start the Backend API
```bash
python3 app.py
```
This starts the FastAPI server on **http://localhost:8080**. It automatically connects to the MCP server and discovers all available tools.

### Step 5: Start the Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Access the interface at **http://localhost:3000**.

---

## 🔌 Adding New Tools (The MCP Advantage)

The biggest benefit of this architecture: **adding a new tool requires zero changes to the agent code**.

Just add a new `@mcp.tool()` function in `mcp_server.py`:

```python
@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for real-time information."""
    # your implementation here
    return results
```

Restart the MCP server, and the agent will **automatically discover and use the new tool** — no changes needed in `app.py` or `main.py`.

---

## 📚 Data Indexing Operations

To populate the vector database with your documents:

1.  Place your target PDF in the root folder (e.g., `fake_company.pdf`).
2.  Open `index_data.py` and ensure the `pdf_path` matches your target file.
3.  Execute the indexer:
    ```bash
    python3 index_data.py
    ```

To perform a hard reset and wipe the database:
```bash
python3 reset_db.py
```
