"""
Unit tests for model-selection scoring and the eval runner.

The scorer is pure and deterministic, so the ship decision is testable
without the target machine: the RAM ceiling is a hard gate, then accuracy
> RAM headroom > latency decide among survivors.
"""

from __future__ import annotations

import pytest
from app.config.constants import RAM_CEILING_MB
from app.evaluation import evaluate
from app.profiling.model_selection import CandidateMetrics, select_best

pytestmark = pytest.mark.unit


def _cand(model_id, ram, acc=0.8, halluc=1.0, p50=200.0, p95=400.0):
    return CandidateMetrics(
        model_id=model_id,
        peak_ram_mb=ram,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        accuracy_overall=acc,
        hallucination_pass_rate=halluc,
    )


def test_over_ceiling_is_disqualified():
    over = _cand("big", RAM_CEILING_MB + 1000)
    under = _cand("small", 3000)
    result = select_best([over, under])
    assert result.winner_id == "small"
    ranks = {r.metrics.model_id: r for r in result.ranked}
    assert ranks["big"].disqualified is True
    assert "DISQUALIFIED" in " ".join(ranks["big"].reasons)


def test_accuracy_preferred_when_budget_and_latency_equal():
    a = _cand("acc-high", 3000, acc=0.9)
    b = _cand("acc-low", 3000, acc=0.6)
    result = select_best([a, b])
    assert result.winner_id == "acc-high"


def test_more_headroom_wins_on_accuracy_tie():
    lean = _cand("lean", 3000, acc=0.8)
    heavy = _cand("heavy", 6000, acc=0.8)  # under ceiling but near warn
    result = select_best([lean, heavy])
    assert result.winner_id == "lean"


def test_latency_breaks_ties():
    fast = _cand("fast", 3000, acc=0.8, p50=100)
    slow = _cand("slow", 3000, acc=0.8, p50=900)
    result = select_best([fast, slow])
    assert result.winner_id == "fast"


def test_all_over_budget_no_winner():
    result = select_best(
        [
            _cand("a", RAM_CEILING_MB + 100),
            _cand("b", RAM_CEILING_MB + 500),
        ]
    )
    assert result.winner_id is None
    assert "No candidate" in result.rationale


def test_effective_accuracy_folds_hallucination():
    honest = CandidateMetrics(
        "h", 3000, 200, 400, accuracy_overall=1.0, hallucination_pass_rate=1.0
    )
    liar = CandidateMetrics("l", 3000, 200, 400, accuracy_overall=1.0, hallucination_pass_rate=0.0)
    assert honest.effective_accuracy == 1.0
    assert liar.effective_accuracy == pytest.approx(0.8)
    # The honest model scores higher, all else equal.
    assert select_best([honest, liar]).winner_id == "h"


def test_selection_result_serializable():
    result = select_best([_cand("m", 3000)])
    d = result.as_dict()
    assert d["winner"] == "m"
    assert d["ram_ceiling_mb"] == RAM_CEILING_MB
    assert d["candidates"][0]["model_id"] == "m"

    # ---------------------------------------------------------------------------
    # Eval runner
    # ---------------------------------------------------------------------------


def test_runner_produces_report(fake_llm):
    from app.orchestration.dispatcher import Dispatcher
    from app.verification.verifier import Verifier

    dispatcher = Dispatcher(llm_client=fake_llm, retriever=None)
    report = evaluate(
        dispatcher,
        fake_llm,
        Verifier,
        config={
            "prompt_version": "v1",
            "model_id": "m",
            "swahili_enrich": True,
            "verification_enabled": True,
        },
    )
    assert report["summary"]["cases"] >= 6
    assert "hallucination_pass_rate" in report["summary"]
    assert report["config"]["model_id"] == "m"
