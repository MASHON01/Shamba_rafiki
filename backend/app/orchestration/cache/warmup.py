"""
Startup warm-up.

The first generation after llama-server starts is the slow one: the model
has to become fully resident and its KV cache is cold. If that cost lands
on the first farmer of the day, the kiosk feels broken. So at startup we
fire one tiny throwaway generation to pay that cost up front, before
anyone is waiting.

It is deliberately tolerant: if llama-server isn't up yet, warm-up just
returns False and the app carries on (the model can warm on the first
real query instead). It never raises and never blocks startup on a dead
server for long - the client's short connect timeout sees to that.

Related warm-path notes (no code, but part of the deliverable):
  - Prompt-prefix reuse. Our prompt always puts the *stable* system
    prompt first, then context, then the question. Keeping the long,
    unchanging prefix at the front is what lets llama-server reuse its KV
    cache across requests - so this ordering is intentional, not
    incidental.
  - Embedding cache. Retrieval already reuses content-
    addressed EmbeddingCache (app/retrieval/embeddings/cache.py), so a
    repeated query doesn't re-embed. Confirmed reused; nothing to change.
"""

from __future__ import annotations

from app.config.constants import WARMUP_MAX_TOKENS
from app.orchestration.llm.base import BaseLLMClient
from app.orchestration.llm.generation_config import GenerationConfig
from app.orchestration.llm.llama_client import LLMError
from app.orchestration.prompts.builder import BuiltPrompt
from app.utils.logger import get_logger

logger = get_logger("Warmup")

_WARMUP_PROMPT = BuiltPrompt(
    system_prompt="You are a helpful assistant.",
    user_prompt="Question: hello",
    full_prompt="You are a helpful assistant.\n\nQuestion: hello\nAnswer:",
    history=[],
)


def warmup(llm: BaseLLMClient) -> bool:
    """
    Run one tiny generation to make the model resident and warm the KV
    cache. Returns True on success, False if the model isn't reachable
    (in which case it simply warms on the first real query instead).
    """
    config = GenerationConfig(max_tokens=WARMUP_MAX_TOKENS, temperature=0.0)
    try:
        llm.generate(_WARMUP_PROMPT, config)
        logger.info("warmup.ok")
        return True
    except LLMError as exc:
        logger.info("warmup.skipped", reason=str(exc))
        return False
    except Exception as exc:  # noqa: BLE001 - warm-up must never break startup.
        logger.warning("warmup.error", reason=str(exc))
        return False
