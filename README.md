# RAG Intelligence: Enterprise AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FastAPI**, **ChromaDB**, and **Next.js**. This project allows you to index your own PDF/Web documents and query them securely using either entirely local models or lightning-fast cloud APIs.

---

## 🏗 Tech Stack

*   **LLM Providers:** [Ollama](https://ollama.com/) (Local) OR [OpenRouter](https://openrouter.ai/) (Cloud API)
*   **Embeddings:** Nomic Embed Text (Local via Ollama)
*   **AI Logic:** [LangChain](https://www.langchain.com/) & LangGraph (ReAct Agent)
*   **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Running via Docker)
*   **Backend API:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **Frontend UI:** [Next.js](https://nextjs.org/) (React, Tailwind CSS, Framer Motion)

---

## 📂 Project Structure

```text
RAG/
├── docker-compose.yml       # Sets up the ChromaDB container
├── requirements.txt         # Python dependencies
├── app.py                   # FastAPI Server (The main Backend)
├── index_data.py            # Script to load, chunk & index PDFs into ChromaDB
├── reset_db.py              # Script to wipe the database clean
│
├── rag/                     # Core LangChain Logic
│   ├── document_loader.py   # Extracts text from PDFs & Web URLs
│   ├── indexing.py          # Chunks text & stores vectors
│   ├── retrieval.py         # The tool the AI uses to search the DB
│   ├── generator.py         # LangGraph ReAct Agent setup
│   ├── db_connection.py     # Connects to ChromaDB
│   └── api_models.py        # Cloud Model Configuration (OpenRouter)
│
└── frontend/                # Next.js UI Application
    ├── src/app/page.tsx     # Main chat interface
    └── src/app/globals.css  # UI Styling (Apple/Google design principles)
```

---

## 🛠 Prerequisites

Before running the project, ensure you have the following installed:
1.  **Python 3.10+** (with a virtual environment set up)
2.  **Node.js 18+** (for the Next.js frontend)
3.  **Docker** (to run ChromaDB)
4.  **Ollama** (installed locally for embeddings & optional local generation).

**Pull the required Ollama models:**
```bash
ollama pull llama3.1
ollama pull nomic-embed-text-v2-moe
```

*(Note: If you plan to use OpenRouter for generation, ensure you add your `OPENROUTER_API_KEY` to `rag/api_models.py`)*

---

## 🚀 Execution Steps

You will need three terminal windows to run the different parts of the stack.

### Step 1: Initialize the Database
Open terminal 1, navigate to the root directory, and start ChromaDB:
```bash
docker compose up -d chromadb
```

### Step 2: Initialize the Backend API
Open terminal 2, activate your python virtual environment, and run:
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Step 3: Initialize the Frontend UI
Open terminal 3, navigate to the `frontend` folder, and run:
```bash
cd frontend
npm install
npm run dev
```
You can now access the interface at **http://localhost:3000**.

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
