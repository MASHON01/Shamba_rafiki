"""Unit tests for retrieval: embedding cache, vector stores, retriever."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.models.document import DocumentChunk, EmbeddedChunk
from app.retrieval.embeddings import EmbeddingCache
from app.retrieval.store import NumpyStore, create_vector_store, load_store, save_store
from app.retrieval.store.persistence import VectorStorePersistenceError

pytestmark = pytest.mark.unit


def _embedded(text: str, vec: list[float], **meta) -> EmbeddedChunk:
    chunk = DocumentChunk(
        document_id=uuid4(), chunk_index=0, text=text,
        token_count=len(text.split()),
        metadata={k: str(v) for k, v in meta.items()},
    )
    return EmbeddedChunk(chunk=chunk, embedding=vec)


# --- embedding cache ------------------------------------------------------

def test_cache_miss_then_hit(tmp_path: Path):
    cache = EmbeddingCache(model_id="m", dimension=3, cache_dir=tmp_path)
    assert cache.get("maize") is None
    cache.set("maize", [1.0, 2.0, 3.0])
    assert cache.get("maize") == [1.0, 2.0, 3.0]
    assert cache.stats == {"hits": 1, "misses": 1}


def test_cache_model_isolation(tmp_path: Path):
    a = EmbeddingCache(model_id="A", dimension=3, cache_dir=tmp_path)
    b = EmbeddingCache(model_id="B", dimension=3, cache_dir=tmp_path)
    a.set("x", [1.0, 1.0, 1.0])
    assert b.get("x") is None  # different model -> no collision


def test_cache_corrupt_entry_is_miss(tmp_path: Path):
    cache = EmbeddingCache(model_id="m", dimension=3, cache_dir=tmp_path)
    entry = cache._entry_path("tomato")
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("{not json", encoding="utf-8")
    assert cache.get("tomato") is None
    assert not entry.exists()  # purged


# --- numpy store ----------------------------------------------------------

def test_numpy_store_ranks_by_similarity():
    store = NumpyStore(4)
    store.add([
        _embedded("maize", [1, 0, 0, 0], crop="maize"),
        _embedded("beans", [0, 1, 0, 0], crop="beans"),
    ])
    results = store.search([1, 0, 0, 0], top_k=2)
    assert results[0].chunk.text == "maize"
    assert results[0].similarity_score >= results[1].similarity_score


def test_numpy_store_threshold_filters():
    store = NumpyStore(4)
    store.add([_embedded("a", [1, 0, 0, 0]), _embedded("b", [0, 1, 0, 0])])
    results = store.search([1, 0, 0, 0], top_k=2, score_threshold=0.5)
    assert all(r.similarity_score >= 0.5 for r in results)


def test_numpy_store_rejects_wrong_dimension():
    store = NumpyStore(4)
    with pytest.raises(ValueError):
        store.add([_embedded("bad", [1, 0])])


def test_numpy_store_empty_search():
    assert NumpyStore(4).search([1, 0, 0, 0]) == []


# --- backend parity + persistence -----------------------------------------

def test_numpy_faiss_parity():
    corpus = [
        _embedded("maize", [1.0, 0.0, 0.0, 0.0]),
        _embedded("beans", [0.0, 1.0, 0.0, 0.0]),
        _embedded("mix", [0.7, 0.3, 0.0, 0.0]),
    ]
    nps = create_vector_store(4, backend="numpy")
    fas = create_vector_store(4, backend="faiss")
    nps.add(corpus); fas.add(corpus)
    q = [0.8, 0.2, 0.0, 0.0]
    np_rank = [r.chunk.text for r in nps.search(q, top_k=3)]
    fa_rank = [r.chunk.text for r in fas.search(q, top_k=3)]
    assert np_rank == fa_rank


def test_persistence_round_trip(tmp_path: Path):
    store = create_vector_store(4, backend="numpy")
    store.add([_embedded("maize blight", [1, 0, 0, 0], crop="maize")])
    save_store(store, backend="numpy", store_dir=tmp_path)

    reloaded = load_store(create_vector_store(4, backend="numpy"),
                          backend="numpy", store_dir=tmp_path)
    assert len(reloaded) == 1
    res = reloaded.search([1, 0, 0, 0], top_k=1)
    assert res[0].chunk.metadata.get("crop") == "maize"


def test_persistence_rejects_dimension_mismatch(tmp_path: Path):
    store = create_vector_store(4, backend="numpy")
    store.add([_embedded("x", [1, 0, 0, 0])])
    save_store(store, backend="numpy", store_dir=tmp_path)
    with pytest.raises(VectorStorePersistenceError):
        load_store(create_vector_store(8, backend="numpy"),
                   backend="numpy", store_dir=tmp_path)


# --- retriever (uses the shared fixture) ----------------------------------

def test_retriever_finds_relevant_chunk(retriever):
    hits = retriever.retrieve("maize blight fungicide", top_k=3)
    assert len(hits) >= 1
    assert "maize" in hits[0].chunk.text.lower()


def test_retriever_empty_query(retriever):
    assert retriever.retrieve("") == []