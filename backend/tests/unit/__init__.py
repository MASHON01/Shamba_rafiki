"""
Unit tests: fast, isolated component tests.

Each module tests one subsystem in isolation (ingestion, retrieval,
language, prompts, verification, logging) with no model weights, no
llama-server, and no network - using the deterministic fakes from
conftest.py where a component boundary needs them.
"""