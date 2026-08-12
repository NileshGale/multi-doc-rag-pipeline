"""
app.py — FastAPI backend for the RAG Q&A interface
---------------------------------------------------
Loads the RAG pipeline on startup, then serves:
  POST /ask        { "question": "...", "session_id": "..." }  → { "answer": "...", "sources": [...] }
  POST /ask/stream { "question": "...", "session_id": "..." }  → SSE token streaming
  GET  /           → serves ask_que.html
  GET  /health     → system status & stats

Run:
    py -3.13 -m uvicorn app:app --reload
Then open: http://localhost:8000
"""

import asyncio
from collections import defaultdict
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

# Disable CUDA before torch loads — CUDA context is unavailable in FastAPI subprocess on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import chromadb
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uuid

load_dotenv()

# ───────────────────────── Logging setup ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [RAG] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_app")

# ───────────────────────── Config ─────────────────────────
PDF_FOLDER       = os.getenv("PDF_FOLDER", "data/pdfs")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "pdf_documents")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K            = int(os.getenv("TOP_K", "7"))
SCORE_THRESHOLD  = float(os.getenv("SCORE_THRESHOLD", "0.35"))
ALLOWED_ORIGINS  = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
CACHE_TTL_SEC    = int(os.getenv("CACHE_TTL_SEC", "3600"))  # 1 hour default TTL
# ──────────────────────────────────────────────────────────

# Rate Limiter setup (15 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address)


# ───────────────────── Custom Exception Classes ───────────────────
class RAGPipelineError(Exception):
    """Transport-agnostic custom domain exception for pipeline & LLM generation errors."""
    pass


# ───────────────────── Cache & Conversation Memory ─────────────────
class CacheItem:
    def __init__(self, answer: str, docs: List[dict], ttl_seconds: int = CACHE_TTL_SEC):
        self.answer = answer
        self.docs = docs
        self.timestamp = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds


class QueryCache:
    """In-memory TTL cache for query responses."""
    def __init__(self, max_size: int = 100, ttl_seconds: int = CACHE_TTL_SEC):
        self.cache: Dict[str, CacheItem] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def get(self, query: str) -> Optional[Tuple[str, List[dict]]]:
        key = query.strip().lower()
        item = self.cache.get(key)
        if not item:
            return None
        if item.is_expired():
            del self.cache[key]
            return None
        return item.answer, item.docs

    def set(self, query: str, answer: str, docs: List[dict]):
        key = query.strip().lower()
        if len(self.cache) >= self.max_size:
            # Evict oldest entry (FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = CacheItem(answer, docs, self.ttl_seconds)


class ConversationSessionStore:
    """Session-based multi-turn chat history store."""
    def __init__(self, max_history_turns: int = 5):
        # Maps session_id -> list of (user_question, bot_answer)
        self.sessions: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.max_history_turns = max_history_turns

    def get_history(self, session_id: Optional[str]) -> List[Tuple[str, str]]:
        if not session_id:
            return []
        return self.sessions[session_id][-self.max_history_turns:]

    def add_turn(self, session_id: Optional[str], question: str, answer: str):
        if not session_id:
            return
        self.sessions[session_id].append((question, answer))
        # Trim history if exceeding max limit
        if len(self.sessions[session_id]) > self.max_history_turns * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_turns:]


query_cache = QueryCache(max_size=100, ttl_seconds=CACHE_TTL_SEC)
session_store = ConversationSessionStore(max_history_turns=5)


# ───────────────────── Pipeline classes ───────────────────

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name} (device=cpu)")
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
        logger.info(f"Vector store ready — {self.collection.count()} chunks indexed")

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
        logger.info(f"Indexed {len(docs_content)} chunks")


class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store      = vector_store

    def retrieve(self, query, top_k=TOP_K, score_threshold=SCORE_THRESHOLD):
        q_emb   = self.embedding_manager.generate_embeddings([query])[0]
        # Query more candidates to allow deduplication of repeated chunks
        fetch_k = max(top_k * 4, 20)
        results = self.vector_store.collection.query(
            query_embeddings=[q_emb.tolist()], n_results=fetch_k
        )
        retrieved = []
        seen_contents = set()

        if results["documents"] and results["documents"][0]:
            for doc_id, meta, doc, dist in zip(
                results["ids"][0], results["metadatas"][0],
                results["documents"][0], results["distances"][0],
            ):
                sim = 1 - dist
                normalized_doc = doc.strip()
                if sim >= score_threshold and normalized_doc not in seen_contents:
                    seen_contents.add(normalized_doc)
                    retrieved.append({
                        "document": doc,
                        "source":   meta.get("source", ""),
                        "page":     meta.get("page", ""),
                        "similarity_score": round(sim, 3),
                    })
                    if len(retrieved) >= top_k:
                        break
        return retrieved


def build_prompt(query: str, docs: List[dict], history: List[Tuple[str, str]] = None) -> Tuple[str, List[dict]]:
    """Construct prompt context from retrieved documents and conversation memory history."""
    context = "\n\n".join(d["document"] for d in docs) if docs else ""
    if not context:
        return "", []

    history_str = ""
    if history:
        turns = []
        for q, a in history:
            turns.append(f"User: {q}\nAssistant: {a}")
        history_str = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

    prompt = f"""You are a retrieval-augmented question answering assistant.
Answer the question using ONLY the provided context and conversation history.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not present say: "I could not find the answer in the provided documents."
4. Give a concise and direct answer.
5. if user ask code then provide the valid code of that programming language
6. if user want example or code then provide the right code and explanation



{history_str}Context:
{context}

Question:
{query}

Answer:"""
    return prompt, docs


def generate_output(query: str, retriever: RAGRetriever, llm: ChatOpenAI, top_k=TOP_K, history: List[Tuple[str, str]] = None) -> Tuple[str, List[dict]]:
    """Executes RAG retrieval + LLM generation, raising transport-agnostic RAGPipelineError on failure."""
    docs = retriever.retrieve(query, top_k)
    prompt, filtered_docs = build_prompt(query, docs, history)

    if not prompt:
        return "I could not find relevant information in the provided documents.", []

    try:
        response = llm.invoke(prompt)
        answer = response.content
        return answer, filtered_docs
    except Exception as e:
        logger.error(f"LLM call failed for query '{query}': {str(e)}", exc_info=True)
        raise RAGPipelineError(
            "The AI generation service is temporarily unavailable. Please try again shortly."
        ) from e


# ─────────────────────── App lifecycle ────────────────────

pipeline: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once on startup with Groq / NVIDIA LLM support."""
    groq_key   = os.getenv("GROQ_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")

    if groq_key:
        logger.info("Initializing LLM provider: Groq (llama-3.1-8b-instant)")
        llm = ChatOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024,
        )
    elif nvidia_key:
        logger.info("Initializing LLM provider: NVIDIA NIM (meta/llama-3.1-70b-instruct)")
        llm = ChatOpenAI(
            api_key=nvidia_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model="meta/llama-3.1-70b-instruct",
            temperature=0.1,
            max_tokens=1024,
        )
    else:
        raise RuntimeError("Missing API key: Set GROQ_API_KEY or NVIDIA_API_KEY in .env")

    em = EmbeddingManager()
    vs = VectorStoreManager()

    # Build index only if empty (first run)
    if vs.collection.count() == 0:
        if not os.path.isdir(PDF_FOLDER):
            raise RuntimeError(
                f"PDF folder not found: '{PDF_FOLDER}'. Create it and place PDF files inside."
            )
        logger.info("Building vector store from PDFs …")
        all_docs = []
        for fn in sorted(os.listdir(PDF_FOLDER)):
            if fn.lower().endswith(".pdf"):
                loader = PyMuPDF4LLMLoader(os.path.join(PDF_FOLDER, fn))
                all_docs.extend(loader.load())
        if not all_docs:
            raise RuntimeError(
                f"No PDF files found in '{PDF_FOLDER}'. Add at least one PDF file."
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks     = splitter.split_documents(all_docs)
        embeddings = em.generate_embeddings([c.page_content for c in chunks])
        vs.add_documents(chunks, embeddings)

    retriever = RAGRetriever(em, vs)
    pipeline["retriever"] = retriever
    pipeline["llm"]       = llm
    logger.info("Pipeline ready -- OK")
    yield
    pipeline.clear()


app = FastAPI(title="RAG Q&A API", lifespan=lifespan)

# Attach rate limiter & error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────── Routes ───────────────────────────

class Question(BaseModel):
    question: str
    session_id: Optional[str] = None


def format_sources(docs: List[dict]) -> List[dict]:
    sources = []
    seen = set()
    for d in docs:
        key = (d["source"], d["page"])
        if key not in seen:
            seen.add(key)
            src = os.path.basename(d["source"]) if d["source"] else "Unknown"
            sources.append({"file": src, "page": d["page"], "score": d["similarity_score"]})
    return sources


@app.get("/")
async def serve_ui():
    """Serve the chat HTML interface."""
    html_path = Path("ask_que.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="ask_que.html not found")
    return FileResponse(html_path, media_type="text/html")


@app.post("/ask")
@limiter.limit("15/minute")
async def ask(request: Request, body: Question):
    """Answer a question using the RAG pipeline with caching, session memory, logging, and rate limiting."""
    start_time = time.time()
    q = body.question.strip()
    session_id = body.session_id

    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if "retriever" not in pipeline or "llm" not in pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    # 1. Check Cache (only for stateless queries without session history)
    history = session_store.get_history(session_id)
    if not history:
        cached = query_cache.get(q)
        if cached:
            answer, docs = cached
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"[CACHE HIT] query='{q}' | latency={elapsed_ms}ms")
            return JSONResponse({"answer": answer, "sources": format_sources(docs), "cached": True})

    # 2. Run Retrieval + Generation in executor
    loop = asyncio.get_running_loop()
    try:
        answer, docs = await loop.run_in_executor(
            None, generate_output, q, pipeline["retriever"], pipeline["llm"], TOP_K, history
        )
    except RAGPipelineError as err:
        raise HTTPException(status_code=503, detail=str(err))

    # 3. Save to Cache, Update Session Memory & Log
    if not history:
        query_cache.set(q, answer, docs)
    session_store.add_turn(session_id, q, answer)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"[PROCESSED] query='{q}' | session='{session_id}' | retrieved={len(docs)} docs | latency={elapsed_ms}ms")

    return JSONResponse({"answer": answer, "sources": format_sources(docs), "cached": False})


@app.post("/ask/stream")
@limiter.limit("15/minute")
async def ask_stream(request: Request, body: Question):
    """Real-time token streaming response endpoint via SSE (Server-Sent Events) with session memory."""
    q = body.question.strip()
    session_id = body.session_id

    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if "retriever" not in pipeline or "llm" not in pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    retriever: RAGRetriever = pipeline["retriever"]
    llm: ChatOpenAI        = pipeline["llm"]

    # Retrieve history & documents
    history = session_store.get_history(session_id)
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(None, retriever.retrieve, q)
    prompt, filtered_docs = build_prompt(q, docs, history)
    sources = format_sources(filtered_docs)

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        if not prompt:
            fallback = "I could not find relevant information in the provided documents."
            yield f"data: {json.dumps({'type': 'content', 'delta': fallback})}\n\n"
            yield "data: [DONE]\n\n"
            return

        full_answer = []
        try:
            async for chunk in llm.astream(prompt):
                content = chunk.content
                if content:
                    full_answer.append(content)
                    yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
            
            final_text = "".join(full_answer)
            if not history:
                query_cache.set(q, final_text, filtered_docs)
            session_store.add_turn(session_id, q, final_text)

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream generation error for query '{q}': {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Stream disconnected unexpectedly.'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    if "retriever" not in pipeline:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "indexed_chunks": 0},
        )
    return {
        "status": "ok",
        "provider": "Groq" if os.getenv("GROQ_API_KEY") else "NVIDIA",
        "indexed_chunks": pipeline["retriever"].vector_store.collection.count(),
        "cached_queries": len(query_cache.cache),
        "active_sessions": len(session_store.sessions),
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
