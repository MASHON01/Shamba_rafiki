"""
Prompt engine & AI orchestrator (Phase 1, Output 6).

The heart of Farm Pal - the layer that ties language
understanding and retrieval together, assembles a grounded prompt,
and calls the LLM:

    Request -> Language -> Retriever -> Prompt -> LLM -> Verifier

This package owns the middle of that pipeline. Language (Output 5)
and retrieval (Output 4) are consumed as finished components; the
verifier (Output 7) plugs in after generation.

Sub-packages / modules:
    prompts/       system prompts, context assembly, final prompt build
    llm/           llama-server client behind an abstract interface
    memory.py      per-session conversation history
    router.py      decides the pipeline path for a request
    dispatcher.py  executes the selected stages
    orchestrator.py  the public entry point: request -> response
"""

from __future__ import annotations

from app.orchestration.dispatcher import Dispatcher, DispatchResult
from app.orchestration.llm import BaseLLMClient, LlamaClient, LLMError
from app.orchestration.memory import ConversationMemory
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.router import RequestRouter, Route, RoutePlan

__all__ = [
    "Orchestrator",
    "Dispatcher",
    "DispatchResult",
    "ConversationMemory",
    "RequestRouter",
    "Route",
    "RoutePlan",
    "BaseLLMClient",
    "LlamaClient",
    "LLMError",
]