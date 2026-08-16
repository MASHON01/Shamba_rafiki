"""
Prompt-length budgeting.

The whole prompt - system instructions + reference context + prior
conversation + the question - plus room for the model's answer, must fit
inside the model's context window (MODEL_CONTEXT_SIZE). The context block
is already capped upstream (ContextBuilder), and the question and system
prompt are fixed, so the one elastic part is the conversation history.

This budgeter fits the history to whatever room is left, dropping the
OLDEST turns first. That is the right thing to lose: the current
question and the most recent exchange matter most; a turn from earlier
in the visit matters least. If nothing else fits, history is dropped
entirely - the question is never sacrificed to keep old chatter.

Token counts use the same pessimistic char/token estimate as the context
builder (CHARS_PER_TOKEN_ESTIMATE), plus a safety margin, so a slight
underestimate does not overflow the real tokenizer at generation time.
No dependency on builder.py - turns are duck-typed (``.question`` /
``.answer``) to avoid an import cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar

from app.config.constants import (
    CHARS_PER_TOKEN_ESTIMATE,
    PROMPT_ANSWER_RESERVE_TOKENS,
    PROMPT_SAFETY_MARGIN_TOKENS,
)
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger("PromptBudgeter")


class _Turn(Protocol):
    question: str
    answer: str


T = TypeVar("T", bound=_Turn)


@dataclass(slots=True)
class BudgetReport:
    """What the budgeter did, for logging and tests."""

    window_tokens: int
    reserve_tokens: int
    fixed_tokens: int  # system + question + margin (non-history)
    history_tokens: int  # estimate for the history that was kept
    kept_turns: int
    dropped_turns: int
    fits: bool  # did even the fixed part fit the budget?


def estimate_tokens(text: str) -> int:
    """
    Pessimistic token estimate for ``text``.

    Deliberately rounds up (ceil) and uses the same low char/token ratio
    as the context builder, so the estimate over-counts rather than
    under-counts - overflowing the window is the failure we guard against.
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def _render_turn(turn: _Turn) -> str:
    # Mirror how the builder flattens a turn, so the estimate matches
    # what actually goes to the model.
    return f"Question: {turn.question.strip()}\nAnswer: {turn.answer.strip()}\n"


def fit_history(
    history: Sequence[T],
    fixed_text: str,
    window_tokens: int | None = None,
    answer_reserve_tokens: int = PROMPT_ANSWER_RESERVE_TOKENS,
    safety_margin_tokens: int = PROMPT_SAFETY_MARGIN_TOKENS,
) -> tuple[list[T], BudgetReport]:
    """
    Trim ``history`` so the whole prompt fits the context window.

    Parameters
    ----------
    history:
        Prior turns, oldest first.
    fixed_text:
        The non-history prompt text (system prompt + reference context +
        question), used to size what room is left for history.
    window_tokens:
        Context window. Defaults to ``settings.MODEL_CONTEXT_SIZE``.
    answer_reserve_tokens, safety_margin_tokens:
        Room held back for the answer and for estimate error.

    Returns
    -------
    (kept_history, report)
        ``kept_history`` keeps the most recent turns that fit, in the
        original oldest-first order.
    """
    window = window_tokens if window_tokens is not None else settings.MODEL_CONTEXT_SIZE

    fixed_tokens = estimate_tokens(fixed_text) + safety_margin_tokens
    history_budget = window - answer_reserve_tokens - fixed_tokens

    # Walk newest -> oldest, keeping turns while they fit; drop the rest.
    kept_reversed: list[T] = []
    used = 0
    for turn in reversed(history):
        cost = estimate_tokens(_render_turn(turn))
        if used + cost > history_budget:
            break
        kept_reversed.append(turn)
        used += cost

    kept = list(reversed(kept_reversed))
    dropped = len(history) - len(kept)

    report = BudgetReport(
        window_tokens=window,
        reserve_tokens=answer_reserve_tokens,
        fixed_tokens=fixed_tokens,
        history_tokens=used,
        kept_turns=len(kept),
        dropped_turns=dropped,
        fits=history_budget >= 0,
    )

    if dropped:
        logger.debug(
            "prompt.history_trimmed",
            kept=len(kept),
            dropped=dropped,
            history_budget_tokens=history_budget,
        )
    if not report.fits:
        # The fixed part alone already exceeds the window. History is
        # fully dropped (kept is empty); the caller still gets a prompt,
        # but this is worth surfacing - the context cap may be too high
        # for this model's window.
        logger.warning(
            "prompt.window_overflow",
            window_tokens=window,
            fixed_tokens=fixed_tokens,
            reserve_tokens=answer_reserve_tokens,
        )

    return kept, report
