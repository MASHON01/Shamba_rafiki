"""
Text chunking.

Splits clean text into retrieval-sized chunks - paragraph-aware,
targeting the ~200-400 token band the architecture calls for, with a
small overlap so a fact split across a boundary still surfaces in
retrieval.

Token counting here is a whitespace-word approximation, deliberately
matching how `pipeline.py` computes `token_count` for each chunk.
The real embedding tokenizer isn't introduced until Output 4; keeping
both on the same simple measure means chunk sizes and reported token
counts stay consistent within Phase 1.

Satisfies the `Chunker` protocol in `app.ingestion.pipeline`
(`chunk(text) -> list[str]`).
"""

from __future__ import annotations

import re

from app.config.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _token_count(text: str) -> int:
    """Whitespace-word count - matches pipeline.py's token_count."""
    return len(text.split())


class Chunker:
    """
    Paragraph-aware chunker with token-based sizing and overlap.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        paragraphs = [
            para.strip()
            for para in _PARAGRAPH_SPLIT_RE.split(text)
            if para.strip()
        ]

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for paragraph in paragraphs:
            para_tokens = _token_count(paragraph)

            # A single paragraph larger than the budget is split on
            # sentence boundaries so we never emit an oversized chunk.
            if para_tokens > self._chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                chunks.extend(self._split_large_paragraph(paragraph))
                continue

            if current_tokens + para_tokens > self._chunk_size and current:
                chunks.append(" ".join(current))
                current, current_tokens = self._carry_overlap(current)

            current.append(paragraph)
            current_tokens += para_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        """Sentence-pack an oversized paragraph into <= chunk_size pieces.

        If a single 'sentence' is itself over the cap (long tables,
        bullet lists, or prose with no . ! ? punctuation), fall back to
        a hard word-window split so no oversized chunk is ever emitted."""
        sentences = [
            s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()
        ]

        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _token_count(sentence)

            # A lone over-cap sentence can't be packed - hard-split it.
            if sent_tokens > self._chunk_size:
                if current:
                    pieces.append(" ".join(current))
                    current, current_tokens = [], 0
                pieces.extend(self._hard_split_words(sentence))
                continue

            if current_tokens + sent_tokens > self._chunk_size and current:
                pieces.append(" ".join(current))
                current, current_tokens = self._carry_overlap(current)

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            pieces.append(" ".join(current))

        return pieces

    def _hard_split_words(self, text: str) -> list[str]:
        """
        Last-resort split on word boundaries into <= chunk_size windows
        with overlap. Used only when sentence splitting can't get a
        piece under the cap.
        """
        words = text.split()
        step = self._chunk_size - self._chunk_overlap  # > 0, enforced in __init__

        windows: list[str] = []
        start = 0
        while start < len(words):
            window = words[start : start + self._chunk_size]
            windows.append(" ".join(window))
            start += step

        return windows

    def _carry_overlap(self, previous_units: list[str]) -> tuple[list[str], int]:
        """
        Seed the next chunk with a trailing slice of the previous one,
        up to chunk_overlap tokens, taken from the end backwards.
        """
        if self._chunk_overlap <= 0:
            return [], 0

        carried: list[str] = []
        carried_tokens = 0

        for unit in reversed(previous_units):
            unit_tokens = _token_count(unit)
            if carried_tokens + unit_tokens > self._chunk_overlap:
                break
            carried.insert(0, unit)
            carried_tokens += unit_tokens

        return carried, carried_tokens