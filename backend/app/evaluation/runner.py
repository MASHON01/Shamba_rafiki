"""
Evaluation runner.

Runs the golden set + hallucination probes through a prepared pipeline
(dispatcher -> generate -> verify -> grade) and returns a config-tagged
report. Factored out so every caller measures accuracy the same way -
the run_eval CLI, the model-selection harness, and the config tuner all
call this, rather than each re-implementing the loop.

It takes the pipeline pieces (dispatcher, llm, verifier) rather than a
whole orchestrator so a harness can point it at any candidate model or
config without rebuilding the app.
"""

from __future__ import annotations

from app.evaluation.golden_set import golden_cases
from app.evaluation.grader import AnswerGrader
from app.evaluation.hallucination_probes import HALLUCINATION_PROBES, grade_probe
from app.evaluation.report import build_report
from app.orchestration.router import RequestRouter


def evaluate(dispatcher, llm, verifier, config: dict, grader: AnswerGrader | None = None) -> dict:
    """
    Score a configuration on the golden set + probes.

    Parameters
    ----------
    dispatcher:
        Provides ``prepare`` (language -> retrieval -> prompt -> config).
    llm:
        The client to generate with (any BaseLLMClient / chain).
    verifier:
        The Verifier applied to each answer.
    config:
        Metadata describing this run (prompt version, model, toggles),
        recorded in the report so runs are comparable.

    Returns
    -------
    dict
        The report from ``build_report`` (summary + per-case + probes).
    """
    grader = grader or AnswerGrader
    plan = RequestRouter.route_text

    grades = []
    for case in golden_cases:
        prepared = dispatcher.prepare(query=case.question, plan=plan, language_hint=case.language)
        generation = llm.generate(prepared.prompt, prepared.config)
        verified = verifier.verify(
            generation.text, prepared.context.sources, language=prepared.language
        )
        grades.append(grader.grade(case, verified.text, prepared.context.sources, verified))

    probe_results = []
    for probe in HALLUCINATION_PROBES:
        prepared = dispatcher.prepare(query=probe.question, plan=plan, language_hint=probe.language)
        generation = llm.generate(prepared.prompt, prepared.config)
        verified = verifier.verify(
            generation.text, prepared.context.sources, language=prepared.language
        )
        probe_results.append(grade_probe(probe, verified))

    return build_report(config, grades, probe_results)
