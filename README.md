# Multi-Document RAG (Retrieval-Augmented Generation) Pipeline & Web Interface

An end-to-end Python Retrieval-Augmented Generation (RAG) system built using **LangChain**, **PyMuPDF4LLM (`langchain-pymupdf4llm`)**, **Sentence-Transformers (`all-MiniLM-L6-v2`)**, **ChromaDB**, **FastAPI**, **Groq API (`llama-3.1-8b-instant`)**, and **NVIDIA NIM API (`meta/llama-3.1-70b-instruct`)**.

Includes a modern, interactive web application (**Lumina AI**) built with custom glassmorphism styling, real-time SSE token streaming, ambient glow animations, structured feature shortcuts, and grounded source attributions.

---

## 📌 Project Overview

This project implements a complete multi-document RAG architecture capable of ingesting PDF documents, chunking text recursively, generating dense 384-dimensional vector embeddings, indexing them into a persistent vector database with Cosine Similarity search, and serving an interactive Q&A web interface grounded using ultra-fast LLMs (Groq / NVIDIA).

### 🔑 Key Features
- **Standalone PDF Ingestion**: High-quality document parsing using `langchain-pymupdf4llm` (`PyMuPDF4LLMLoader`).
- **Text Splitting & Chunking**: Recursive character text splitting (`chunk_size=500`, `chunk_overlap=50`) preserving document boundary semantics.
- **Dense Vector Embeddings**: HuggingFace `SentenceTransformer` using model `all-MiniLM-L6-v2` (384 dimensions).
- **Persistent Vector Store**: ChromaDB integration with `hnsw:space = cosine` index metric and metadata tracking (`source`, `page`, `doc_index`).
- **Semantic Retrieval**: Custom `RAGRetriever` class with threshold-filtered cosine similarity scoring (`SCORE_THRESHOLD=0.4`, `TOP_K=5`).
- **Groq & NVIDIA LLM Engine**: Ultra-fast LLM inference using **Groq (`llama-3.1-8b-instant`)** with automatic fallback to **NVIDIA NIM (`llama-3.1-70b-instruct`)**.
- **Real-Time Token Streaming**: Server-Sent Events (SSE) streaming (`POST /ask/stream`) powering a ChatGPT-style typing experience.
- **In-Memory Query Caching**: In-memory cache for duplicate queries delivering sub-5ms responses with zero API costs.
- **Rate Limiting (`slowapi`)**: Built-in rate limiting (15 requests/min) protecting server compute and LLM API costs.
- **Robust Error Handling**: Exception boundaries wrapping LLM calls, returning clean HTTP 503 responses instead of raw 500 server crashes.
- **Structured Logging**: Request profiling logging timestamps, client IPs, retrieval stats, cache hits, and latency (ms).
- **Modern Responsive UI (Lumina AI)**: Redesigned web interface (`ask_que.html`) with pastel glassmorphism accents, live token streaming, prompt cards, and source attribution badges.

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
                         │ Groq / NVIDIA 70B LLM        │ ───► Real-Time SSE Stream & Sources
                         └──────────────────────────────┘
```

---

## 📂 Directory Structure

```
.
├── app.py                   # FastAPI backend with Groq/NVIDIA LLM, SSE streaming, rate limiting, logging & caching
├── ask_que.html             # Lumina AI frontend with real-time token streaming and glassmorphism styling
├── RAG_pipeline.ipynb       # Jupyter Notebook detailing individual pipeline steps & evaluation
├── data/
│   ├── pdfs/                # Directory containing PDF research documents
│   └── vector_store/        # Persistent ChromaDB database store (ignored in git)
├── .env                     # Local environment variables (GROQ_API_KEY, NVIDIA_API_KEY)
├── .gitignore               # Ignores secrets, temporary caches, and vector DB
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A **Groq API Key** (from [Groq Console](https://console.groq.com/)) OR an **NVIDIA API Key** (from [NVIDIA Build](https://build.nvidia.com/))

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
   pip install fastapi uvicorn langchain-pymupdf4llm langchain-text-splitters sentence-transformers chromadb langchain-openai python-dotenv requests slowapi
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY="gsk_your-actual-groq-api-key-here"
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
*(Or run with `uvicorn app:app --reload`)*

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

### API Endpoints
- **`GET /`**: Serves the Lumina AI Q&A web interface (`ask_que.html`).
- **`POST /ask`**: Synchronous Q&A endpoint. Accepts `{"question": "..."}` and returns `{"answer": "...", "sources": [...]}`.
- **`POST /ask/stream`**: Server-Sent Events (SSE) token-by-token streaming endpoint for real-time text generation.
- **`GET /health`**: Returns system status, active LLM provider, cached queries count, and total indexed vector chunk count.

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).

