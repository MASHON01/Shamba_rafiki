"""
Abstract LLM client interface.

Every LLM backend - the llama-server client here, or anything that
replaces it later - implements this. The orchestrator depends only on
this interface, never on a concrete client, so the backend can change
without touching the pipeline.

Two methods, matching the two things the orchestrator needs:

    generate(prompt)  run inference, return the answer text
    health()          is the backend reachable and ready?

`GenerationResult` carries the text plus light metadata (token counts,
latency) the orchestrator threads into ProcessingMetadata for the
speed/efficiency scoring the build plan cares about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.orchestration.prompts.builder import BuiltPrompt


@dataclass(slots=True)
class GenerationResult:
    """
    Output of a single LLM generation.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None


class BaseLLMClient(ABC):
    """
    Base class for LLM backends.
    """

    @abstractmethod
    def generate(self, prompt: BuiltPrompt) -> GenerationResult:
        """
        Run inference for `prompt` and return the generated answer.

        Raises
        ------
        app.core.exceptions.ShambaRafikiError (or a subclass)
            On connection failure, timeout, or a bad response - never
            a raw library exception, so the orchestrator only handles
            one error family.
        """
        raise NotImplementedError

    @abstractmethod
    def health(self) -> bool:
        """
        Return True if the backend is reachable and ready to serve.
        Never raises - a failure is reported as False.
        """
        raise NotImplementedError