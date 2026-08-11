# Multi-Document RAG (Retrieval-Augmented Generation) Pipeline & Web Interface

An end-to-end Python Retrieval-Augmented Generation (RAG) system built using **LangChain**, **PyMuPDF4LLM (`langchain-pymupdf4llm`)**, **Sentence-Transformers (`all-MiniLM-L6-v2`)**, **ChromaDB**, **FastAPI**, and **NVIDIA NIM API (`meta/llama-3.1-70b-instruct`)**.

Includes a modern, interactive web application (**Lumina AI**) built with custom glassmorphism styling, ambient glow animations, structured feature shortcuts, and real-time document grounding.

---

## 📌 Project Overview

This project implements a complete multi-document RAG architecture capable of ingesting PDF documents, chunking text recursively, generating dense 384-dimensional vector embeddings, indexing them into a persistent vector database with Cosine Similarity search, and serving an interactive Q&A web interface grounded using NVIDIA's hosted Llama 3.1 70B Instruct model.

### 🔑 Key Features
- **Standalone PDF Ingestion**: High-quality document parsing using `langchain-pymupdf4llm` (`PyMuPDF4LLMLoader`).
- **Text Splitting & Chunking**: Recursive character text splitting (`chunk_size=500`, `chunk_overlap=50`) preserving document boundary semantics.
- **Dense Vector Embeddings**: HuggingFace `SentenceTransformer` using model `all-MiniLM-L6-v2` (384 dimensions).
- **Persistent Vector Store**: ChromaDB integration with `hnsw:space = cosine` index metric and metadata tracking (`source`, `page`, `doc_index`).
- **Semantic Retrieval**: Custom `RAGRetriever` class with threshold-filtered cosine similarity scoring (`SCORE_THRESHOLD=0.4`, `TOP_K=5`).
- **Grounded LLM Generation**: Prompt-engineered QA pipeline powered by **NVIDIA NIM API (Llama 3.1 70B)** with strict hallucination-prevention constraints.
- **FastAPI Web Backend**: Async web backend (`app.py`) providing `/ask` and `/health` REST API endpoints alongside static UI serving.
- **Modern Responsive UI (Lumina AI)**: Redesigned web interface (`ask_que.html`) with pastel glassmorphism accents (`#FFE2E2`, `#A2CB8B`, `#9AD872`, `#CFECF3`, `#F9D0CD`), glowing animated hero graphics, Font Awesome iconography, prompt cards, and source attribution badges.

---

## 🏗️ System Architecture

```
┌─────────────────┐      ┌──────────────────────────────┐      ┌───────────────────────────┐
│ PDF Documents   │ ───► │ PyMuPDF4LLMLoader Ingestion │ ───► │ Text Splitting (500 chars)│
└─────────────────┘      └──────────────────────────────┘      └───────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────┐      ┌──────────────────────────────┐      ┌───────────────────────────┐
│ User Query      │ ───► │ SentenceTransformer          │ ◄─── │ SentenceTransformer       │
│ (FastAPI /ask)  │      │ (all-MiniLM-L6-v2)           │      │ Embeddings (384-dim)      │
└─────────────────┘      └──────────────────────────────┘      └───────────────────────────┘
         │                               │                                   │
         │                               ▼                                   ▼
         │               ┌──────────────────────────────┐      ┌───────────────────────────┐
         └─────────────► │ Semantic Search (Chroma)     │ ◄─── │ ChromaDB Persistent Store │
                         └──────────────────────────────┘      └───────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ NVIDIA Llama 3.1 70B LLM     │ ───► Grounded Response & Sources
                         └──────────────────────────────┘
```

---

## 📂 Directory Structure

```
.
├── app.py                   # FastAPI backend server with lifespan lifecycle & RAG pipeline
├── ask_que.html             # Modern Lumina AI frontend UI with glassmorphism styling
├── RAG_pipeline.ipynb       # Jupyter Notebook detailing individual pipeline steps & evaluation
├── data/
│   ├── pdfs/                # Directory containing PDF research documents
│   └── vector_store/        # Persistent ChromaDB database store (ignored in git)
├── .env                     # Local environment variables (NVIDIA_API_KEY)
├── .gitignore               # Ignores secrets, temporary caches, and vector DB
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- An active **NVIDIA API Key** (from [NVIDIA Build](https://build.nvidia.com/))

### 2. Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install fastapi uvicorn langchain-pymupdf4llm langchain-text-splitters sentence-transformers chromadb langchain-openai python-dotenv requests
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   NVIDIA_API_KEY="nvapi-your-actual-nvidia-api-key-here"
   ```

5. **Upload PDF Documents**:
   Place your PDF files into the `data/pdfs/` directory:
   ```bash
   mkdir -p data/pdfs
   ```
   > 📄 Any `.pdf` document placed inside `data/pdfs/` will be automatically chunked and indexed into ChromaDB when starting `app.py`.

---

## 🖥️ Running the Application

Start the FastAPI application:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

### API Endpoints
- **`GET /`**: Serves the Lumina AI Q&A web interface (`ask_que.html`).
- **`POST /ask`**: Accepts `{"question": "..."}` and returns `{"answer": "...", "sources": [...]}`.
- **`GET /health`**: Returns system status and total indexed vector chunk count.

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
