"""
Model selection scoring.

Given the measured numbers for each candidate GGUF - peak RAM, latency,
accuracy - pick the one to ship, on evidence rather than vibes. The
scoring encodes this project's priorities, in order:

  1. Stay under the 7 GB ceiling. This is a hard gate, not a weight:
     a candidate whose peak RAM exceeds the ceiling is DISQUALIFIED
     outright, because exceeding it risks an OOM kill - a total failure.
  2. Accuracy is the rubric core. It carries the most weight among the
     survivors (answer quality + honest uncertainty).
  3. RAM headroom beats raw speed. On an 8 GB machine, headroom is what
     wins the Efficiency score and the Budget Profile bonus, so it
     outweighs latency.
  4. Latency breaks near-ties.

Pure and deterministic, so the choice is reproducible and testable; the
select_model.py harness feeds it the measured numbers from the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.constants import RAM_CEILING_MB, RAM_WARN_MB

# Weights among candidates that pass the RAM gate. Accuracy first, then
# headroom, then latency - matching the priorities above.
DEFAULT_WEIGHTS = {"accuracy": 0.5, "ram_headroom": 0.3, "latency": 0.2}


@dataclass(slots=True)
class CandidateMetrics:
    """Measured numbers for one candidate model (from the target)."""

    model_id: str
    peak_ram_mb: float
    latency_p50_ms: float
    latency_p95_ms: float
    accuracy_overall: float  # 0-1 (golden-set mean)
    hallucination_pass_rate: float  # 0-1
    latency_drift_pct: float | None = None  # thermal/sustained-load drift
    notes: str = ""

    @property
    def effective_accuracy(self) -> float:
        # Honest-uncertainty behaviour is part of accuracy for this
        # product (the hidden-prompt scenario), so fold it internally.
        return 0.8 * self.accuracy_overall + 0.2 * self.hallucination_pass_rate

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "accuracy_overall": round(self.accuracy_overall, 4),
            "hallucination_pass_rate": round(self.hallucination_pass_rate, 4),
            "effective_accuracy": round(self.effective_accuracy, 4),
            "latency_drift_pct": self.latency_drift_pct,
            "notes": self.notes,
        }


@dataclass(slots=True)
class RankedCandidate:
    """A candidate with its computed score and gate status."""

    metrics: CandidateMetrics
    score: float
    within_budget: bool
    disqualified: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            **self.metrics.as_dict(),
            "score": round(self.score, 4),
            "within_budget": self.within_budget,
            "disqualified": self.disqualified,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class SelectionResult:
    """The ranked field, the winner, and why."""

    ranked: list[RankedCandidate]
    winner_id: str | None
    rationale: str

    def as_dict(self) -> dict:
        return {
            "winner": self.winner_id,
            "rationale": self.rationale,
            "candidates": [r.as_dict() for r in self.ranked],
            "ram_ceiling_mb": RAM_CEILING_MB,
        }


def select_best(
    candidates: list[CandidateMetrics],
    ram_ceiling_mb: int = RAM_CEILING_MB,
    ram_warn_mb: int = RAM_WARN_MB,
    weights: dict | None = None,
) -> SelectionResult:
    """
    Score and rank candidates, returning the winner + rationale.

    Over-ceiling candidates are disqualified (never the winner) but still
    scored and listed for transparency. Latency is normalized across the
    field (best p50 = 1.0, worst = 0.0), so it only ever breaks ties.
    """
    weights = weights or DEFAULT_WEIGHTS
    if not candidates:
        return SelectionResult(ranked=[], winner_id=None, rationale="No candidates.")

    p50s = [c.latency_p50_ms for c in candidates]
    lo, hi = min(p50s), max(p50s)

    ranked: list[RankedCandidate] = []
    for c in candidates:
        within = c.peak_ram_mb <= ram_ceiling_mb
        headroom = max(0.0, min(1.0, (ram_ceiling_mb - c.peak_ram_mb) / ram_ceiling_mb))
        latency_score = (
            1.0 if hi == lo else max(0.0, min(1.0, 1 - (c.latency_p50_ms - lo) / (hi - lo)))
        )
        score = (
            weights["accuracy"] * c.effective_accuracy
            + weights["ram_headroom"] * headroom
            + weights["latency"] * latency_score
        )

        reasons: list[str] = []
        if not within:
            reasons.append(
                f"DISQUALIFIED: peak {c.peak_ram_mb:.0f} MB exceeds the "
                f"{ram_ceiling_mb} MB ceiling (OOM risk)."
            )
        elif c.peak_ram_mb >= ram_warn_mb:
            reasons.append(
                f"within budget but near the warn line ({c.peak_ram_mb:.0f} "
                f">= {ram_warn_mb} MB) - little headroom."
            )
        if c.latency_drift_pct is not None and c.latency_drift_pct > 20:
            reasons.append(
                f"latency drifted {c.latency_drift_pct:.0f}% under sustained "
                f"load - possible thermal throttling."
            )

        ranked.append(
            RankedCandidate(
                metrics=c,
                score=score,
                within_budget=within,
                disqualified=not within,
                reasons=reasons,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    eligible = [r for r in ranked if not r.disqualified]
    winner = eligible[0] if eligible else None

    return SelectionResult(
        ranked=ranked,
        winner_id=winner.metrics.model_id if winner else None,
        rationale=_rationale(winner, ranked, ram_ceiling_mb),
    )


def _rationale(
    winner: RankedCandidate | None,
    ranked: list[RankedCandidate],
    ram_ceiling_mb: int,
) -> str:
    if winner is None:
        return (
            "No candidate stayed under the RAM ceiling - none is shippable "
            "as measured. Try a smaller model or a lighter quantization."
        )
    m = winner.metrics
    headroom = ram_ceiling_mb - m.peak_ram_mb
    parts = [
        f"Selected {m.model_id}: peak RAM {m.peak_ram_mb:.0f} MB leaves "
        f"{headroom:.0f} MB headroom under the {ram_ceiling_mb} MB ceiling, "
        f"accuracy {m.accuracy_overall:.0%} (hallucination honesty "
        f"{m.hallucination_pass_rate:.0%}), p50 latency "
        f"{m.latency_p50_ms:.0f} ms.",
    ]
    disq = [r for r in ranked if r.disqualified]
    if disq:
        parts.append(
            "Disqualified for exceeding the ceiling: "
            + ", ".join(r.metrics.model_id for r in disq)
            + "."
        )
    return " ".join(parts)
