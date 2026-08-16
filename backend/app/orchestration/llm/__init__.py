"""
LLM backends.

    BaseLLMClient    the interface (generate / health)
    GenerationResult text + token/latency metadata
    LlamaClient      llama.cpp llama-server HTTP client

The orchestrator depends only on BaseLLMClient, so the concrete
backend can be swapped without touching the pipeline.
"""

from __future__ import annotations

from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.llama_client import (
    LlamaClient,
    LLMConnectionError,
    LLMError,
)

__all__ = [
    "BaseLLMClient",
    "GenerationResult",
    "LlamaClient",
    "LLMError",
    "LLMConnectionError",
]