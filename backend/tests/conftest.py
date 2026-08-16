"""
Shared pytest fixtures for the Shamba Rafiki backend suite.

Centralizes the test doubles and setup that the ad-hoc scripts used
to copy-paste:

- deterministic FakeEmbedder / FakeLLM so tests run with no model
  weights, no llama-server, and fully predictable output,
- a temp-corpus builder that runs the real ingestion + indexing
  pipeline into a throwaway directory,
- a ready Retriever over that corpus,
- a TestClient with dependency overrides wiring the fake LLM + real
  retriever behind the real FastAPI app.

Everything here is real application code EXCEPT the embedder and LLM,
which are the only two components that need actual model weights.
That keeps the tests honest (they exercise the true pipeline) while
staying fast and offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.retrieval.embeddings.base import BaseEmbedder


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------

# A small fixed vocabulary the fake embedder projects text onto, so
# topically-similar strings land near each other and retrieval is
# genuinely meaningful (not random) without a real model.
_FAKE_VOCAB = [
    "maize", "beans", "tomato", "cassava", "blight", "rust", "wilt",
    "fungicide", "mancozeb", "rotation", "nakuru", "kiambu", "price", "treat",
]


class FakeEmbedder(BaseEmbedder):
    """Deterministic bag-of-words embedder over a fixed vocabulary."""

    def __init__(self) -> None:
        self.dimension = len(_FAKE_VOCAB)

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        # +0.01 base so no vector is all-zero (cosine stays defined).
        return [float(low.count(word)) + 0.01 for word in _FAKE_VOCAB]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM(BaseLLMClient):
    """LLM stub returning a fixed, corpus-grounded answer."""

    def __init__(self, answer: str | None = None, healthy: bool = True) -> None:
        self._answer = answer or (
            "Apply mancozeb fungicide at 40 grams per 20 litres of water "
            "every 7 days, and rotate maize with beans."
        )
        self._healthy = healthy
        self.last_prompt = None
        self.calls = 0

    def generate(self, prompt) -> GenerationResult:
        self.calls += 1
        self.last_prompt = prompt
        return GenerationResult(
            text=self._answer,
            prompt_tokens=120,
            completion_tokens=24,
            latency_ms=35,
        )

    def health(self) -> bool:
        return self._healthy


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_corpus_text() -> str:
    return (
        "Maize leaf blight is a fungal disease common in Nakuru county. "
        "Apply mancozeb fungicide at 40 grams per 20 litres of water. "
        "Spray every 7 days. Crop rotation with beans reduces blight in "
        "following seasons. "
    ) * 3


@pytest.fixture
def temp_corpus_dir(tmp_path: Path, sample_corpus_text: str) -> Path:
    """
    Run the REAL ingestion pipeline over a small doc into tmp_path, and
    return the processed-corpus directory (containing manifest.json).
    """
    from app.ingestion.pipeline import IngestionPipeline
    from app.ingestion.builder import CorpusBuilder
    from app.ingestion.extractors import TextExtractor
    from app.ingestion.processors import (
        Chunker, Cleaner, DuplicateDetector, Hasher, MetadataGenerator,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "maize_guide.txt").write_text(sample_corpus_text, encoding="utf-8")

    corpus = tmp_path / "processed"
    pipeline = IngestionPipeline(
        extractor=TextExtractor(), cleaner=Cleaner(), chunker=Chunker(),
        metadata_generator=MetadataGenerator(), hasher=Hasher(),
        duplicate_detector=DuplicateDetector(corpus_dir=corpus),
        builder=CorpusBuilder(output_dir=corpus),
    )
    pipeline.run(raw, source="KALRO", language="en")
    return corpus


@pytest.fixture
def retriever(temp_corpus_dir: Path, tmp_path: Path, fake_embedder: FakeEmbedder):
    """
    A real Retriever over a real index built from the temp corpus, using
    the fake embedder. Lower default threshold because the toy embedder
    produces lower cosines than a real MiniLM model.
    """
    from app.retrieval.indexer import Indexer
    from app.retrieval.search import Retriever
    from app.retrieval.store import create_vector_store

    store_dir = tmp_path / "vector_store"
    Indexer(
        embedder=fake_embedder,
        store=create_vector_store(fake_embedder.dimension, backend="numpy"),
        backend="numpy",
    ).build(corpus_dir=temp_corpus_dir, store_dir=store_dir)

    return Retriever.from_disk(
        store_dir=store_dir, backend="numpy",
        embedder=FakeEmbedder(), default_threshold=0.1,
    )


@pytest.fixture
def orchestrator(retriever, fake_llm: FakeLLM):
    """A full orchestrator wired with the fake LLM and real retriever."""
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    return Orchestrator(
        llm_client=fake_llm,
        dispatcher=Dispatcher(llm_client=fake_llm, retriever=retriever),
    )


@pytest.fixture
def api_client(retriever, fake_llm: FakeLLM):
    """
    A TestClient over the real FastAPI app, with the orchestrator and
    retriever dependencies overridden to use the fake LLM + real
    retriever. No llama-server required.
    """
    from fastapi.testclient import TestClient

    from app.api.app_factory import create_app
    from app.api.dependencies import get_orchestrator, get_retriever
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    test_orch = Orchestrator(
        llm_client=fake_llm,
        dispatcher=Dispatcher(llm_client=fake_llm, retriever=retriever),
    )

    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: test_orch
    app.dependency_overrides[get_retriever] = lambda: retriever

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()

# --------------------------------------------------------------------------
# Vision / image-classifier fixtures
# --------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_classifier_path() -> Path:
    return FIXTURES_DIR / "tiny_classifier.onnx"


@pytest.fixture
def leaf_image(tmp_path: Path) -> Path:
    import numpy as np
    from PIL import Image

    path = tmp_path / "leaf.jpg"
    pixels = (np.random.default_rng(1).random((240, 320, 3)) * 255).astype("uint8")
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture
def image_classifier(tiny_classifier_path: Path):
    from app.vision import ImageClassifier

    return ImageClassifier(model_path=tiny_classifier_path)


@pytest.fixture
def image_orchestrator(retriever, fake_llm, image_classifier):
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    return Orchestrator(
        llm_client=fake_llm,
        dispatcher=Dispatcher(llm_client=fake_llm, retriever=retriever),
        classifier=image_classifier,
    )
