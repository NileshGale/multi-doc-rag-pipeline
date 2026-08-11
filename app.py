"""
app.py — FastAPI backend for the RAG Q&A interface
---------------------------------------------------
Loads the RAG pipeline on startup, then serves:
  POST /ask   { "question": "..." }  → { "answer": "...", "sources": [...] }
  GET  /      → serves ask_que.html

Run:
    py -3.13 -m uvicorn app:app --reload
Then open: http://localhost:8000
"""

import asyncio
import os
# Disable CUDA before torch loads — CUDA context is unavailable in FastAPI subprocess on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from langchain_openai import ChatOpenAI

load_dotenv()

# ───────────────────────── Config ─────────────────────────
PDF_FOLDER       = "data/pdfs"
VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME  = "pdf_documents"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
TOP_K            = 5
SCORE_THRESHOLD  = 0.4
# ──────────────────────────────────────────────────────────


# ───────────────────── Pipeline classes ───────────────────

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"[startup] Loading embedding model: {model_name} (device=cpu)")
        # Force CPU — CUDA context is unavailable in FastAPI subprocess
        self.model = SentenceTransformer(model_name, device="cpu")

    def generate_embeddings(self, text):
        return self.model.encode(text, show_progress_bar=False, device="cpu")


class VectorStoreManager:
    def __init__(self, persist_directory=VECTOR_STORE_DIR, collection_name=COLLECTION_NAME):
        self.persist_directory = persist_directory
        self.collection_name   = collection_name
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG pdf embeddings", "hnsw:space": "cosine"},
        )
        print(f"[startup] Vector store ready — {self.collection.count()} chunks indexed")

    def add_documents(self, documents, embeddings):
        ids, docs_content, embeds_list, metadatas = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4()}")
            meta = dict(doc.metadata)
            meta["doc_index"] = i
            metadatas.append(meta)
            docs_content.append(doc.page_content)
            embeds_list.append(emb.tolist())
        self.collection.add(ids=ids, metadatas=metadatas,
                            documents=docs_content, embeddings=embeds_list)
        print(f"[startup] Indexed {len(docs_content)} chunks")


class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store      = vector_store

    def retrieve(self, query, top_k=TOP_K, score_threshold=SCORE_THRESHOLD):
        q_emb   = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[q_emb.tolist()], n_results=top_k
        )
        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc_id, meta, doc, dist in zip(
                results["ids"][0], results["metadatas"][0],
                results["documents"][0], results["distances"][0],
            ):
                sim = 1 - dist
                if sim >= score_threshold:
                    retrieved.append({
                        "document": doc,
                        "source":   meta.get("source", ""),
                        "page":     meta.get("page", ""),
                        "similarity_score": round(sim, 3),
                    })
        return retrieved


def generate_output(query, retriever, llm, top_k=TOP_K):
    docs    = retriever.retrieve(query, top_k)
    context = "\n\n".join(d["document"] for d in docs) if docs else ""

    if not context:
        return "I could not find relevant information in the provided documents.", []

    prompt = f"""You are a retrieval-augmented question answering assistant.
Answer the question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not present say: "I could not find the answer in the provided documents."
4. Give a concise and direct answer.

Context:
{context}

Question:
{query}

Answer:"""

    answer = llm.invoke(prompt).content
    return answer, docs


# ─────────────────────── App lifecycle ────────────────────

pipeline: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once on startup."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY missing from .env")

    em  = EmbeddingManager()
    vs  = VectorStoreManager()

    # Build index only if empty (first run)
    if vs.collection.count() == 0:
        if not os.path.isdir(PDF_FOLDER):
            raise RuntimeError(
                f"PDF folder not found: '{PDF_FOLDER}'. "
                "Create it and place your PDF files inside."
            )
        print("[startup] Building vector store from PDFs …")
        all_docs = []
        for fn in sorted(os.listdir(PDF_FOLDER)):
            if fn.lower().endswith(".pdf"):
                loader = PyMuPDF4LLMLoader(os.path.join(PDF_FOLDER, fn))
                all_docs.extend(loader.load())
        if not all_docs:
            raise RuntimeError(
                f"No PDF files found in '{PDF_FOLDER}'. "
                "Add at least one PDF to build the vector store."
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks     = splitter.split_documents(all_docs)
        embeddings = em.generate_embeddings([c.page_content for c in chunks])
        vs.add_documents(chunks, embeddings)

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        model="meta/llama-3.1-70b-instruct",
        temperature=0.1,
        max_tokens=1024,
    )

    retriever = RAGRetriever(em, vs)
    pipeline["retriever"] = retriever
    pipeline["llm"]       = llm
    print("[startup] Pipeline ready -- OK")
    yield
    pipeline.clear()


app = FastAPI(title="RAG Q&A API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────── Routes ───────────────────────────

class Question(BaseModel):
    question: str


@app.get("/")
async def serve_ui():
    """Serve the chat HTML interface."""
    html_path = Path("ask_que.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="ask_que.html not found")
    return FileResponse(html_path, media_type="text/html")


@app.post("/ask")
async def ask(body: Question):
    """Answer a question using the RAG pipeline."""
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if "retriever" not in pipeline or "llm" not in pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    # Run the blocking LLM/retrieval call in a thread pool so the async
    # event loop is not blocked.
    loop = asyncio.get_running_loop()
    answer, docs = await loop.run_in_executor(
        None, generate_output, q, pipeline["retriever"], pipeline["llm"]
    )

    sources = []
    seen    = set()
    for d in docs:
        key = (d["source"], d["page"])
        if key not in seen:
            seen.add(key)
            src = os.path.basename(d["source"]) if d["source"] else "Unknown"
            sources.append({"file": src, "page": d["page"],
                            "score": d["similarity_score"]})

    return JSONResponse({"answer": answer, "sources": sources})


@app.get("/health")
async def health():
    if "retriever" not in pipeline:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "indexed_chunks": 0},
        )
    return {"status": "ok", "indexed_chunks": pipeline["retriever"].vector_store.collection.count()}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
