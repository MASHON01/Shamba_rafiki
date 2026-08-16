"""
Text cleaning.

Removes the noise that PDF/DOCX extraction leaves behind - page
numbers, repeated headers/footers, hyphenation at line breaks, and
irregular whitespace - so the chunker works on clean prose. Runs
*after* extraction, *before* chunking.

Deliberately conservative: it strips obvious boilerplate, not
content. Over-aggressive cleaning silently deletes real agricultural
guidance, which is worse than leaving a stray page number in.

Satisfies the `Cleaner` protocol in `app.ingestion.pipeline`
(`clean(text) -> str`).
"""

from __future__ import annotations

import re
from collections import Counter

# Lines that are just a page number ("12", "- 12 -", "Page 12", "12/40").
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?[-–—]?\s*\d+\s*(?:[-–—/]\s*\d+)?\s*[-–—]?\s*$",
    re.IGNORECASE,
)

# A word split across a line break by hyphenation: "fer-\ntilizer".
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\n(\w+)")

# 3+ blank lines collapse to a paragraph break.
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Runs of spaces/tabs collapse to one space.
_EXCESS_SPACES_RE = re.compile(r"[ \t]{2,}")

# Common OCR/PDF junk: form-feed, zero-width, and non-breaking spaces.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]")

# A line must repeat on at least this fraction of pages to count as a
# running header/footer worth stripping.
_HEADER_FOOTER_MIN_RATIO = 0.5

# Only short lines are header/footer candidates; long lines are prose.
_HEADER_FOOTER_MAX_WORDS = 12


class Cleaner:
    """
    Cleans raw extracted text into normalized prose.
    """

    def clean(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        text = _CONTROL_CHARS_RE.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Rejoin hyphenated line breaks before any other line handling.
        text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

        text = self._strip_repeated_headers_footers(text)

        cleaned_lines = [
            line
            for line in (raw.strip() for raw in text.split("\n"))
            if not _PAGE_NUMBER_RE.match(line)
        ]
        text = "\n".join(cleaned_lines)

        text = _EXCESS_SPACES_RE.sub(" ", text)
        text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)

        return text.strip()

    def _strip_repeated_headers_footers(self, text: str) -> str:
        """
        Detect and drop running headers/footers: short lines that
        repeat across many page breaks. Page breaks come through
        extraction as form-feeds (from PyMuPDF's "\\n\\n" joins we
        approximate with blank-line-delimited blocks).

        We only remove a line if it's short AND repeats on >= half the
        blocks - a strong signal it's boilerplate, not content.
        """
        # Split into page-ish blocks on 2+ newlines.
        blocks = re.split(r"\n{2,}", text)
        if len(blocks) < 4:
            # Too few pages to reliably tell boilerplate from content.
            return text

        candidate_counter: Counter[str] = Counter()
        for block in blocks:
            block_lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if not block_lines:
                continue
            # Header/footer candidates: the top two and bottom two lines
            # of each block. Two (not one) because a footer often sits
            # just above a varying page number, and a header just below
            # a chapter title.
            edge_lines = set(block_lines[:2]) | set(block_lines[-2:])
            for candidate in edge_lines:
                if len(candidate.split()) <= _HEADER_FOOTER_MAX_WORDS:
                    candidate_counter[candidate] += 1

        threshold = max(2, int(len(blocks) * _HEADER_FOOTER_MIN_RATIO))
        boilerplate = {
            line for line, count in candidate_counter.items()
            if count >= threshold
        }

        if not boilerplate:
            return text

        kept_lines = [
            line
            for line in text.split("\n")
            if line.strip() not in boilerplate
        ]
        return "\n".join(kept_lines)