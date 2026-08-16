"""Integration test: the corpus build path, ingest -> index -> retrieve.

Exercises the real ingestion pipeline, real indexer, and real retriever
wired together (only the embedder is faked), proving a document put in
one end is retrievable by meaning out the other."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.builder import MANIFEST_FILENAME

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_ingestion_produces_manifest_and_chunks(temp_corpus_dir: Path):
    manifest_path = temp_corpus_dir / MANIFEST_FILENAME
    assert manifest_path.exists(), "ingestion must write a manifest"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # manifest is a dict with a cumulative documents list + counts
    assert manifest["document_count"] >= 1
    assert manifest["chunk_count"] >= 1
    assert len(manifest["documents"]) >= 1

    # each listed document has a matching record file under documents/
    docs_dir = temp_corpus_dir / "documents"
    for entry in manifest["documents"]:
        assert (docs_dir / f"{entry['document_id']}.json").exists()


def test_chunks_carry_domain_metadata(temp_corpus_dir: Path):
    records = list((temp_corpus_dir / "documents").glob("*.json"))
    assert records
    record = json.loads(records[0].read_text(encoding="utf-8"))
    chunks = record["chunks"]
    assert len(chunks) >= 1
    # our sample text is about maize in Nakuru
    all_meta = [c["metadata"] for c in chunks]
    assert any(m.get("crop") == "maize" for m in all_meta)
    assert any(m.get("county") == "Nakuru" for m in all_meta)


def test_index_build_then_retrieve(retriever):
    assert len(retriever) >= 1, "index should contain chunks"

    hits = retriever.retrieve("maize blight fungicide treatment", top_k=3)
    assert len(hits) >= 1
    top = hits[0]
    assert "maize" in top.chunk.text.lower()
    assert top.similarity_score > 0.0
    assert top.chunk.metadata.get("crop") == "maize"


def test_retrieval_ranking_is_ordered(retriever):
    hits = retriever.retrieve("maize blight", top_k=5)
    scores = [h.similarity_score for h in hits]
    assert scores == sorted(scores, reverse=True), "results must be ranked"


def test_irrelevant_query_low_or_empty(retriever):
    hits = retriever.retrieve("quantum banking cryptocurrency", top_k=3)
    assert all(h.similarity_score < 0.9 for h in hits)