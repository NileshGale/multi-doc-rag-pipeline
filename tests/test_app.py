"""
test_app.py — Automated Unit Tests for RAG application pipeline
"""

import time
import pytest
from app import QueryCache, ConversationSessionStore, build_prompt, RAGPipelineError


def test_query_cache_set_and_get():
    cache = QueryCache(max_size=10, ttl_seconds=3600)
    docs = [{"document": "test doc", "source": "file.pdf", "page": 0, "similarity_score": 0.95}]
    
    cache.set("What is RAG?", "RAG stands for Retrieval-Augmented Generation.", docs)
    cached = cache.get("What is RAG?")
    
    assert cached is not None
    answer, cached_docs = cached
    assert answer == "RAG stands for Retrieval-Augmented Generation."
    assert len(cached_docs) == 1


def test_query_cache_ttl_expiration():
    # Cache with 1 second TTL
    cache = QueryCache(max_size=10, ttl_seconds=1)
    docs = [{"document": "test doc", "source": "file.pdf", "page": 0, "similarity_score": 0.95}]
    
    cache.set("Short question", "Short answer", docs)
    assert cache.get("Short question") is not None
    
    # Wait for TTL to expire
    time.sleep(1.1)
    assert cache.get("Short question") is None


def test_session_store_multi_turn_history():
    store = ConversationSessionStore(max_history_turns=3)
    session_id = "test_session_123"
    
    assert len(store.get_history(session_id)) == 0
    
    store.add_turn(session_id, "What is Machine Learning?", "Machine Learning is a subset of AI.")
    history = store.get_history(session_id)
    assert len(history) == 1
    assert history[0] == ("What is Machine Learning?", "Machine Learning is a subset of AI.")
    
    store.add_turn(session_id, "Give an example", "Linear Regression is an example.")
    history = store.get_history(session_id)
    assert len(history) == 2


def test_build_prompt_with_history():
    docs = [{"document": "Gradient descent minimizes error.", "source": "ml.pdf", "page": 1, "similarity_score": 0.88}]
    history = [("What is gradient descent?", "It is an optimization algorithm.")]
    
    prompt, filtered_docs = build_prompt("How does it work?", docs, history)
    
    assert "Prior Conversation History:" in prompt
    assert "User: What is gradient descent?" in prompt
    assert "Context:" in prompt
    assert "Gradient descent minimizes error." in prompt
    assert len(filtered_docs) == 1


def test_build_prompt_empty_docs():
    prompt, filtered_docs = build_prompt("Unrelated question", [])
    assert prompt == ""
    assert filtered_docs == []
