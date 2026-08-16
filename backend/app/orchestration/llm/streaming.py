"""
Streaming token path for llama-server.

The blocking client (llama_client.generate) waits for the whole answer
before returning. On a CPU-only kiosk that can be several seconds of a
blank screen. Streaming sends the answer token-by-token as it forms, so
the farmer sees it appear immediately - the cheapest possible win on
perceived latency, even when total generation time is unchanged.

This module is the low-level transport: it POSTs to llama-server's
/completion endpoint with ``stream: true`` and turns the Server-Sent
Events response into a sequence of ``StreamEvent``s (token deltas, then a
final event carrying token counts). It knows nothing about prompts,
verification, or HTTP responses to the frontend - the client wraps it,
the assembler folds it into a full answer, and the SSE route re-emits it
to the browser.

Two robustness features live here:
  - Cancellation. A ``threading.Event`` lets the caller stop mid-stream
    (the farmer navigated away); the loop checks it between tokens and
    closes the connection promptly instead of generating into the void.
  - Typed failures. A connection drop mid-stream raises
    ``LLMConnectionError`` so the caller can finalize whatever partial
    answer it already has, rather than seeing a raw requests exception.

The typed errors are imported lazily inside the function to avoid an
import cycle with llama_client (which imports this module at top level).
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Iterator

_DATA_PREFIX = "data:"
_DONE_SENTINEL = "[DONE]"


@dataclass(slots=True)
class StreamEvent:
    """
    One event from the token stream.

    A normal event carries a ``content`` delta. The final event has
    ``done=True`` and, when llama-server provides them, the prompt/
    completion token counts (only meaningful on that last event).
    """

    content: str = ""
    done: bool = False
    stop_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def iter_completion_stream(
    requests,
    url: str,
    payload: dict,
    timeout: float | tuple[float, float],
    cancel_event=None,
) -> Iterator[StreamEvent]:
    """
    Stream a /completion response as ``StreamEvent``s.

    Parameters
    ----------
    requests:
        The (lazily-imported) requests module, passed in by the client.
    url, payload, timeout:
        The POST target, JSON body (must include ``stream: true``), and
        read timeout.
    cancel_event:
        Optional ``threading.Event``; when set, the stream stops and the
        connection is closed at the next token boundary.

    Yields
    ------
    StreamEvent
        Token deltas as they arrive, then a final ``done`` event.

    Raises
    ------
    LLMConnectionError
        The server was unreachable, timed out, or the connection dropped
        mid-stream.
    LLMError
        A non-200 status or an otherwise malformed response.
    """
    from app.orchestration.llm.llama_client import LLMConnectionError, LLMError

    try:
        response = requests.post(url, json=payload, timeout=timeout, stream=True)
    except requests.exceptions.Timeout as exc:
        raise LLMConnectionError(f"llama-server timed out after {timeout}s at {url}.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise LLMConnectionError(
            f"Could not connect to llama-server at {url}. Is it running?"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"llama-server stream request failed: {exc}") from exc

    if response.status_code != 200:
        _safe_close(response)
        raise LLMError(f"llama-server returned HTTP {response.status_code} on stream.")

    try:
        for raw in response.iter_lines:
            if cancel_event is not None and cancel_event.is_set:
                return  # closed in finally

            if not raw:
                continue  # SSE keep-alive / blank separator line

            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith(_DATA_PREFIX):
                continue

            data = line[len(_DATA_PREFIX) :].strip()
            if data == _DONE_SENTINEL:
                yield StreamEvent(done=True)
                return

            try:
                obj = json.loads(data)
            except ValueError:
                continue  # tolerate a partial/garbled line rather than crash

            event = _event_from_obj(obj)
            yield event
            if event.done:
                return
    except requests.exceptions.RequestException as exc:
        # A drop partway through: signal the caller so it can keep the
        # partial answer assembled so far.
        raise LLMConnectionError(f"llama-server stream interrupted: {exc}") from exc
    finally:
        _safe_close(response)


def _event_from_obj(obj: dict) -> StreamEvent:
    stop = bool(obj.get("stop", False))
    timings = obj.get("timings", {}) or {}
    prompt_tokens = None
    completion_tokens = None
    if stop:
        prompt_tokens = obj.get("tokens_evaluated") or timings.get("prompt_n")
        completion_tokens = obj.get("tokens_predicted") or timings.get("predicted_n")
    return StreamEvent(
        content=obj.get("content", "") or "",
        done=stop,
        stop_reason=obj.get("stop_type"),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _safe_close(response) -> None:
    # Best-effort cleanup; a failed close must never propagate.
    with contextlib.suppress(Exception):
        response.close()
