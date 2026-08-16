"""
llama-server HTTP client.

Talks to a running llama.cpp `llama-server` over its local HTTP API -
the competition's hard requirement (GGUF weights via llama.cpp). The
orchestrator calls `generate()`; this client turns a BuiltPrompt into
a POST to the server's /completion endpoint and returns the answer.

Design choices:
- The model stays resident in llama-server between requests, so this
  client is stateless and cheap - it just sends prompts. That's what
  keeps per-query latency (the speed score) low.
- The HTTP library (`requests`) is imported lazily, so importing this
  module never requires it to be installed - the whole app stays
  importable on a machine that hasn't set up the LLM yet, exactly
  like the embedder's lazy model load.
- Every failure mode - server down, timeout, malformed response - is
  caught and re-raised as a typed LLMError, so the orchestrator never
  sees a raw requests exception.

Uses the /completion endpoint with the flattened `full_prompt`
(rather than /v1/chat/completions) because it's the most universally
available llama-server endpoint and BuiltPrompt already provides the
single-string form. Swapping to the chat endpoint later would only
change this file.
"""

from __future__ import annotations

import time

from app.config.constants import (
    LLM_MAX_TOKENS,
    LLM_STOP_SEQUENCES,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    LLM_TOP_P,
)
from app.config.settings import settings
from app.core.exceptions import ShambaRafikiError
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.prompts.builder import BuiltPrompt
from app.utils.logger import get_logger

logger = get_logger("LlamaClient")

_COMPLETION_PATH = "/completion"
_HEALTH_PATH = "/health"


class LLMError(ShambaRafikiError):
    """Raised when the LLM backend cannot produce a generation."""


class LLMConnectionError(LLMError):
    """Raised when llama-server is unreachable or times out."""


class LlamaClient(BaseLLMClient):
    """
    HTTP client for a local llama.cpp llama-server.
    """

    def __init__(
        self,
        server_url: str | None = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        timeout: int = LLM_TIMEOUT_SECONDS,
        stop: list[str] | None = None,
    ) -> None:
        self._server_url = (server_url or settings.LLM_SERVER_URL).rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._timeout = timeout
        self._stop = stop if stop is not None else list(LLM_STOP_SEQUENCES)

    def generate(self, prompt: BuiltPrompt, config=None) -> GenerationResult:
        requests = self._load_requests()

        payload = {
            "prompt": prompt.full_prompt,
            "n_predict": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "stop": self._stop,
            "stream": False,
        }

        url = f"{self._server_url}{_COMPLETION_PATH}"
        started = time.monotonic()

        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"llama-server timed out after {self._timeout}s at {url}."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to llama-server at {url}. Is it running?"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"llama-server request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"llama-server returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        latency_ms = int((time.monotonic() - started) * 1000)

        return self._parse_response(response, latency_ms)

    def health(self) -> bool:
        try:
            requests = self._load_requests()
        except LLMError:
            return False

        try:
            response = requests.get(
                f"{self._server_url}{_HEALTH_PATH}", timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_response(self, response, latency_ms: int) -> GenerationResult:
        """
        Extract the answer text and token counts from a llama-server
        /completion response. Tolerant of missing timing fields.
        """
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(
                f"llama-server returned non-JSON response: "
                f"{response.text[:200]}"
            ) from exc

        content = data.get("content")
        if content is None:
            raise LLMError(
                "llama-server response had no 'content' field."
            )

        # llama-server nests token counts under 'timings'/'tokens_*'.
        timings = data.get("timings", {}) or {}
        prompt_tokens = data.get("tokens_evaluated") or timings.get(
            "prompt_n"
        )
        completion_tokens = data.get("tokens_predicted") or timings.get(
            "predicted_n"
        )

        text = content.strip()
        logger.info(
            "llm.generated",
            completion_chars=len(text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _load_requests():
        try:
            import requests

            return requests
        except ImportError as exc:
            raise LLMError(
                "The 'requests' library is required to talk to llama-server. "
                "Install it with: pip install requests"
            ) from exc