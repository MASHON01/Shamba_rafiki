"""
Hallucination detector.

Targets the single most dangerous kind of fabrication for a farmer:
an unsupported *specific* - a dose, a price, a percentage, a spacing,
a number of days. A wrong general statement is bad; a confidently
invented "spray 50ml per litre" is the kind of thing someone acts on
and damages a crop or wastes money over.

So this check extracts specific numeric claims from the answer and
verifies each one appears in the retrieved sources. Specifics with no
support are flagged. Non-numeric prose is left to the semantic and
citation checks; this detector deliberately does one narrow, high-
value thing well.

The score is the fraction of specific claims that are supported (1.0
when the answer makes no specific claims at all - nothing risky to
fabricate). Returns a `CheckResult`; unsupported specifics are listed
in `flags` so the policy layer and the operator can see exactly what
was unverified.
"""

from __future__ import annotations

import re

from app.config.constants import SPECIFIC_CLAIM_PATTERN
from app.models.document import RetrievalResult
from app.verification import CheckResult

_SPECIFIC_RE = re.compile(SPECIFIC_CLAIM_PATTERN, re.IGNORECASE)
# Extract just the leading number of each specific, for comparison.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class HallucinationDetector:
    """
    Flags specific numeric claims not supported by the sources.
    """

    def detect(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> CheckResult:
        specifics = self._extract_specifics(answer)

        if not specifics:
            # No numbers/doses/prices to fabricate - lowest-risk case.
            return CheckResult(
                name="hallucination",
                score=1.0,
                passed=True,
                detail="Answer makes no specific numeric claims.",
            )

        source_text = " ".join(r.chunk.text for r in sources).lower()
        source_numbers = set(_NUMBER_RE.findall(source_text))

        unsupported: list[str] = []
        for specific in specifics:
            number = _NUMBER_RE.search(specific)
            if number is None:
                continue
            if number.group() not in source_numbers:
                unsupported.append(specific.strip())

        supported_count = len(specifics) - len(unsupported)
        score = round(supported_count / len(specifics), 3)
        passed = not unsupported

        flags = [f"unsupported_specific:{s}" for s in unsupported]

        return CheckResult(
            name="hallucination",
            score=score,
            passed=passed,
            detail=(
                f"{supported_count}/{len(specifics)} specific claims "
                f"supported by sources."
            ),
            flags=flags,
        )

    def _extract_specifics(self, answer: str) -> list[str]:
        """
        Pull specific numeric claims (with their units) from the
        answer. A bare number with no unit still counts - "spray 3
        times" is a specific worth checking - but pure years or list
        numbering slip through as low-risk; the number-in-source check
        below tolerates that.
        """
        matches = [m.group() for m in _SPECIFIC_RE.finditer(answer)]
        # Keep only matches that actually contain a digit (the pattern's
        # optional unit alone can't match without one, but guard anyway).
        return [m for m in matches if any(c.isdigit() for c in m)]