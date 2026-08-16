"""
Scored evaluation report.

Rolls a run's per-case grades and hallucination-probe results into one
comparable summary, tagged with the configuration that produced it -
prompt version, model, Swahili enrichment, verification on/off. Because
every report carries its config, two runs can be diffed to answer the
only question that matters here: did this change make answers better?

The summary breaks the overall score down by category (standard vs the
self-run hidden prompts), by intent, and by language, plus the
hallucination pass rate - so a regression shows up where it happens
rather than hiding in a single average.
"""

from __future__ import annotations

from statistics import mean
from typing import Iterable

from app.evaluation.grader import GradeResult
from app.evaluation.hallucination_probes import ProbeResult


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(mean(values), 4) if values else 0.0


def _group_mean(grades: list[GradeResult], key) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for g in grades:
        groups.setdefault(key(g), []).append(g.overall)
    return {k: _mean(v) for k, v in sorted(groups.items())}


def build_report(
    config: dict,
    grades: list[GradeResult],
    probe_results: list[ProbeResult],
) -> dict:
    """Aggregate grades + probes into a config-tagged report dict."""
    overall = _mean(g.overall for g in grades)
    pass_rate = round(sum(g.passed for g in grades) / len(grades), 4) if grades else 0.0
    probe_pass = (
        round(sum(p.passed for p in probe_results) / len(probe_results), 4)
        if probe_results
        else 0.0
    )

    return {
        "config": config,
        "summary": {
            "cases": len(grades),
            "overall_mean": overall,
            "pass_rate": pass_rate,
            "grounding_mean": _mean(g.grounding.score for g in grades if g.grounding.applicable),
            "by_category": _group_mean(grades, lambda g: g.category),
            "by_intent": _group_mean(grades, lambda g: g.intent),
            "by_language": _group_mean(grades, lambda g: g.language),
            "hallucination_pass_rate": probe_pass,
            "hallucination_probes": len(probe_results),
        },
        "cases": [
            {
                "id": g.case_id,
                "category": g.category,
                "language": g.language,
                "intent": g.intent,
                "overall": g.overall,
                "passed": g.passed,
                "term_coverage": g.term_coverage,
                "point_coverage": g.point_coverage,
                "grounding_score": round(g.grounding.score, 4),
                "grounded_used": g.grounding.used,
                "language_match": g.language_match,
                "action": g.action,
                "confidence": g.confidence_score,
                "flags": g.flags,
            }
            for g in grades
        ],
        "probes": [
            {
                "id": p.probe_id,
                "passed": p.passed,
                "said_unsure": p.said_unsure,
                "action": p.action,
            }
            for p in probe_results
        ],
    }


def format_report(report: dict) -> str:
    """A compact human-readable rendering of a report."""
    cfg = report["config"]
    s = report["summary"]
    lines = [
        "Answer-quality evaluation",
        "=" * 46,
        f"prompt_version: {cfg.get('prompt_version')}",
        f"model_id: {cfg.get('model_id')}",
        f"swahili_enrich: {cfg.get('swahili_enrich')}",
        f"verification: {cfg.get('verification_enabled')}",
        "-" * 46,
        f"cases: {s['cases']}",
        f"overall mean: {s['overall_mean']:.0%}",
        f"pass rate: {s['pass_rate']:.0%}",
        f"grounding mean: {s['grounding_mean']:.0%}",
        f"hidden prompts: {s['by_category'].get('hidden', 0.0):.0%}",
        f"hallucination: {s['hallucination_pass_rate']:.0%} "
        f"({s['hallucination_probes']} probes)",
        "-" * 46,
        "by intent: " + ", ".join(f"{k}={v:.0%}" for k, v in s["by_intent"].items()),
        "by language: " + ", ".join(f"{k}={v:.0%}" for k, v in s["by_language"].items()),
    ]
    return "\n".join(lines)
