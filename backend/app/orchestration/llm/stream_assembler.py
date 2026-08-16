"""
Stream assembler: token deltas -> one finished answer.

The streaming transport yields many small ``StreamEvent`` deltas; the
rest of the pipeline (verification, memory, the response envelope) wants
one whole answer plus its token/latency metadata, exactly like the
blocking path's ``GenerationResult``. This assembler is the bridge.

It also handles the imperfect case. If the stream ends without a final
``done`` event - a mid-stream connection drop - the caller marks the
assembler interrupted and still calls ``finalize``: the farmer gets the
partial answer that did arrive, flagged as partial, rather than nothing.
"""

from __future__ import annotations

from app.orchestration.llm.base import GenerationResult
from app.orchestration.llm.streaming import StreamEvent


class StreamAssembler:
    """Accumulates ``StreamEvent`` deltas into a full answer."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.done: bool = False
        self.interrupted: bool = False

    def feed(self, event: StreamEvent) -> None:
        """Fold one event into the running answer."""
        if event.content:
            self._parts.append(event.content)
        if event.prompt_tokens is not None:
            self.prompt_tokens = event.prompt_tokens
        if event.completion_tokens is not None:
            self.completion_tokens = event.completion_tokens
        if event.done:
            self.done = True

    def mark_interrupted(self) -> None:
        """Flag that the stream ended early (partial answer)."""
        self.interrupted = True

    @property
    def text(self) -> str:
        """The answer assembled so far, trimmed like the blocking path."""
        return "".join(self._parts).strip()

    @property
    def has_text(self) -> bool:
        return bool(self._parts)

    def finalize(self, latency_ms: int | None = None) -> GenerationResult:
        """
        Package what we have as a ``GenerationResult``.

        If the completion-token count never arrived (interrupted stream),
        fall back to the number of delta pieces received, so downstream
        metadata is still populated with something meaningful.
        """
        completion_tokens = self.completion_tokens
        if completion_tokens is None and self.interrupted:
            completion_tokens = len(self._parts)

        return GenerationResult(
            text=self.text,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
