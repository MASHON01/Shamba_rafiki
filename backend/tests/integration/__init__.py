"""
Integration tests: cross-component flows.

These exercise multiple subsystems wired together against real
application code - the ingest->index->retrieve corpus path and the
full orchestrator pipeline (Request -> Language -> Retrieval ->
Prompt -> LLM -> Verification -> Response) - using the deterministic
fake embedder/LLM from conftest.py so they stay fast and offline.
"""