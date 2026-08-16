"""
Conversation memory.

Per-session history of question/answer turns, so a farmer can ask a
follow-up ("and how much does that cost?") within one visit and the
model sees the prior exchange.

Scoped deliberately for the kiosk model: the terminal is shared and
walk-up (one farmer, then the next), so memory is

- keyed by session_id, kept only in RAM (nothing persisted to disk),
- bounded to the last N turns per session (RAM budget + relevance),
- capped in total sessions, evicting the oldest, so an all-day kiosk
  can't grow memory without limit.

An operator ends a farmer's session (or starts a fresh one) simply by
using a new session_id; `clear(session_id)` is also provided. There
is intentionally no cross-session or cross-visit memory.
"""

from __future__ import annotations

from collections import OrderedDict, deque

from app.config.constants import (
    MEMORY_MAX_SESSIONS,
    MEMORY_MAX_TURNS,
)
from app.orchestration.prompts.builder import ConversationTurn
from app.utils.logger import get_logger

logger = get_logger("ConversationMemory")


class ConversationMemory:
    """
    Bounded, in-RAM, per-session Q/A history.
    """

    def __init__(
        self,
        max_turns: int = MEMORY_MAX_TURNS,
        max_sessions: int = MEMORY_MAX_SESSIONS,
    ) -> None:
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        # session_id -> deque of ConversationTurn (bounded).
        # OrderedDict so we can evict the least-recently-used session.
        self._sessions: OrderedDict[str, deque[ConversationTurn]] = (
            OrderedDict()
        )

    def get_history(self, session_id: str | None) -> list[ConversationTurn]:
        """
        Return the turns for a session, oldest first. Unknown or None
        session -> empty history.
        """
        if not session_id or session_id not in self._sessions:
            return []
        # Touch for LRU recency.
        self._sessions.move_to_end(session_id)
        return list(self._sessions[session_id])

    def add_turn(
        self,
        session_id: str | None,
        question: str,
        answer: str,
    ) -> None:
        """
        Record a completed exchange. No-op if session_id is None
        (stateless request) or either side is empty.
        """
        if not session_id or not question or not answer:
            return

        if session_id not in self._sessions:
            self._evict_if_full()
            self._sessions[session_id] = deque(maxlen=self._max_turns)

        self._sessions[session_id].append(
            ConversationTurn(question=question, answer=answer)
        )
        self._sessions.move_to_end(session_id)

        logger.debug(
            "memory.turn_added",
            session_id=session_id,
            turns=len(self._sessions[session_id]),
        )

    def clear(self, session_id: str | None) -> None:
        """Forget a single session (e.g. operator starts a new farmer)."""
        if session_id and session_id in self._sessions:
            del self._sessions[session_id]

    def clear_all(self) -> None:
        """Forget every session."""
        self._sessions.clear()

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def _evict_if_full(self) -> None:
        """Drop the least-recently-used session when at capacity."""
        while len(self._sessions) >= self._max_sessions:
            evicted, _ = self._sessions.popitem(last=False)
            logger.debug("memory.session_evicted", session_id=evicted)