# Multi-Document RAG (Retrieval-Augmented Generation) Pipeline

An end-to-end Python Retrieval-Augmented Generation (RAG) system built using **LangChain**, **PyMuPDF**, **Sentence-Transformers (`all-MiniLM-L6-v2`)**, **ChromaDB**, and **NVIDIA NIM API (`meta/llama-3.1-70b-instruct`)**.

---

## 📌 Project Overview

This project implements a complete multi-document RAG architecture capable of ingesting PDF documents, chunking text recursively, generating dense 384-dimensional vector embeddings, indexing them into a persistent vector database with Cosine Similarity search, and retrieving relevant context to generate grounded answers using NVIDIA's hosted Llama 3.1 70B Instruct model.

### 🔑 Key Features
- **Bulk PDF Ingestion**: Automatic parsing of multiple PDF files from `data/pdfs/` using `PyMuPDFLoader` (PyMuPDF/fitz).
- **Text Splitting & Chunking**: Recursive character text splitting (`chunk_size=500`, `chunk_overlap=50`) preserving document boundary semantics.
- **Dense Vector Embeddings**: HuggingFace `SentenceTransformer` using model `all-MiniLM-L6-v2` (384 dimensions).
- **Persistent Vector Store**: ChromaDB integration with `hnsw:space = cosine` index metric and document metadata tagging (`doc_id`, `source`, `page`, `content_length`).
- **Semantic Retrieval**: Custom `RAGRetriever` class with threshold-filtered cosine similarity scoring.
- **Grounded LLM Generation**: Prompt-engineered QA pipeline powered by **NVIDIA API (Llama 3.1 70B)** with strict hallucination-prevention constraints.

---

## 🏗️ System Architecture

```
┌─────────────────┐      ┌──────────────────────────┐      ┌───────────────────────────┐
│ PDF Documents   │ ───► │ PyMuPDFLoader Ingestion  │ ───► │ Text Splitting (500 chars)│
└─────────────────┘      └──────────────────────────┘      └───────────────────────────┘
                                                                         │
                                                                         ▼
┌─────────────────┐      ┌──────────────────────────┐      ┌───────────────────────────┐
│ User Query      │ ───► │ SentenceTransformer      │ ◄─── │ SentenceTransformer       │
└─────────────────┘      │ (all-MiniLM-L6-v2)       │      │ Embeddings (384-dim)      │
         │               └──────────────────────────┘      └───────────────────────────┘
         │                             │                                 │
         │                             ▼                                 ▼
         │               ┌──────────────────────────┐      ┌───────────────────────────┐
         └─────────────► │ Semantic Search (Chroma) │ ◄─── │ ChromaDB Persistent Store │
                         └──────────────────────────┘      └───────────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │ NVIDIA Llama 3.1 70B LLM │ ───► Grounded Response
                         └──────────────────────────┘
```

---

## 📂 Directory Structure

```
.
├── RAG_pipeline.ipynb       # Main Jupyter Notebook containing the full RAG pipeline
├── data/
│   ├── pdfs/                # Directory containing PDF research documents (34 files)
│   ├── python.txt           # Text document loader testing file
│   ├── research.pdf         # Single PDF test file
│   └── vector_store/        # Persistent Chroma DB vector database (ignored in git)
├── .env                     # Local environment variables (contains NVIDIA_API_KEY - hidden from git)
├── .gitignore               # Excludes secrets, caches, and vector DB
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed
- Jupyter Notebook / JupyterLab environment
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
   pip install langchain langchain-core langchain-community langchain-openai langchain-text-splitters pypdf pymupdf chromadb sentence-transformers python-dotenv scikit-learn
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   NVIDIA_API_KEY="nvapi-your-actual-nvidia-api-key-here"
   ```

5. **Upload PDF Documents**:
   Create the `data/pdfs/` directory if it doesn't exist, and place your PDF files inside it:
   ```bash
   mkdir -p data/pdfs
   ```
   > 📄 Place any `.pdf` documents you want the RAG system to ingest into the `data/pdfs/` folder. The pipeline will automatically scan, chunk, embed, and index all PDFs placed in this folder.

---

## 🧪 Pipeline Execution Walkthrough

The notebook `RAG_pipeline.ipynb` breaks down into the following key steps:

### Step 1: Document Ingestion
Scans the `data/pdfs/` folder and dynamically loads all PDF documents page-by-page using `PyMuPDFLoader`.

### Step 2: Document Chunking
Splits the 229 loaded pages into **847 text chunks** using `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`.

### Step 3: Embeddings & Vector Storage
Instantiates `EmbeddingManager` (`all-MiniLM-L6-v2`) and initializes `VectorStoreManager` (ChromaDB persistent client). Indexes all 847 chunks into the `pdf_documents` collection with Cosine similarity indexing.

### Step 4: Semantic Retrieval
Configures `RAGRetriever` to embed queries, execute similarity search against Chroma DB, filter results with a `score_threshold` (e.g. 0.4), and return ranked candidate chunks.

### Step 5: Grounded Generation (LLM)
Connects to NVIDIA NIM API hosting `meta/llama-3.1-70b-instruct`. Synthesizes grounded responses with strict instructions:
1. Use **ONLY** provided document context.
2. Do not hallucinate or use external knowledge.
3. Fallback response if answer is absent: `"I could not find the answer in the provided documents."`

---

## 🛠️ Multi-Provider Support

The pipeline includes built-in extensibility sections for alternative LLM providers:
- **NVIDIA NIM API** (`meta/llama-3.1-70b-instruct`) — *Active*
- **OpenAI API** (`gpt-4` / `gpt-3.5-turbo`) — *Optional section*
- **Groq API** (`qwen/qwen3-32b` / `llama3-70b-8192`) — *Optional section*

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
