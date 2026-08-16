"""
Swahili evaluation set.

Real farmer-style questions in Kiswahili, each tagged with the qualities
a good answer must have: the intent, the canonical domain terms the
answer should engage with, and the expectation that the answer comes
back in Swahili. This is what turns "Swahili works" from a claim into a
measured number - the zero-shot baseline (scripts/eval_swahili.py) and
the broader accuracy harness both score against it.

Kept as data, separate from any grader, so the questions can grow
independently of how they're scored. ``expected_terms`` are canonical
English terms (as used in the English corpus and AGRI_TERMS/KNOWN_CROPS);
a good Swahili answer should engage the concept, whether it uses the
English term or its Swahili form - the grader accepts either.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SwahiliEvalCase:
    """One Swahili evaluation question and the qualities a good answer has."""

    id: str
    question: str
    intent: str
    expected_terms: tuple[str, ...] = field(default_factory=tuple)
    crop: str | None = None
    county: str | None = None
    notes: str = ""

    # Spread across intents (diagnosis / how_to / price / general), the four
    # in-scope crops, and the target counties. Expected terms are canonical
    # English concepts a grounded answer should touch.


SWAHILI_EVAL_SET: tuple[SwahiliEvalCase, ...] = (
    SwahiliEvalCase(
        id="sw-diag-maize-blight",
        question=(
            "Mahindi yangu yana madoa ya kahawia kwenye majani na "
            "yananyauka. Tatizo ni nini na nifanyeje?"
        ),
        intent="diagnosis",
        expected_terms=("blight", "maize", "fungicide"),
        crop="maize",
        county="Nakuru",
        notes="Classic maize blight diagnosis; should name the disease and " "give steps.",
    ),
    SwahiliEvalCase(
        id="sw-diag-tomato-blight",
        question="Nyanya zangu zina ukungu kwenye majani, nifanye nini?",
        intent="diagnosis",
        expected_terms=("blight", "tomato", "fungicide"),
        crop="tomato",
        notes="Uses the Swahili term 'ukungu' for blight.",
    ),
    SwahiliEvalCase(
        id="sw-diag-cassava-mosaic",
        question="Muhogo wangu una batobato kwenye majani, ni ugonjwa gani?",
        intent="diagnosis",
        expected_terms=("mosaic", "cassava", "disease"),
        crop="cassava",
        notes="Cassava mosaic ('batobato'); tests less-common crop + term.",
    ),
    SwahiliEvalCase(
        id="sw-pest-maize",
        question=("Mahindi yangu yana wadudu wanaokula majani. Nitumie dawa gani?"),
        intent="diagnosis",
        expected_terms=("pest", "maize", "pesticide"),
        crop="maize",
        notes="Pest control; 'wadudu' -> pest, 'dawa' -> pesticide.",
    ),
    SwahiliEvalCase(
        id="sw-howto-beans-planting",
        question="Nipande maharagwe lini na vipi hapa Kiambu?",
        intent="how_to",
        expected_terms=("beans", "seed"),
        crop="beans",
        county="Kiambu",
        notes="Planting how-to; should give ordered steps and timing.",
    ),
    SwahiliEvalCase(
        id="sw-howto-rotation",
        question="Nizungushe mazao vipi shambani mwangu ili kupunguza magonjwa?",
        intent="how_to",
        expected_terms=("disease", "soil"),
        notes="Crop rotation to reduce disease; general agronomy.",
    ),
    SwahiliEvalCase(
        id="sw-price-beans",
        question="Bei ya maharagwe ikoje Nakuru, na je niuze sasa au nisubiri?",
        intent="price",
        expected_terms=("beans",),
        crop="beans",
        county="Nakuru",
        notes="Market/price + decision framing; must say prices are " "approximate and change.",
    ),
    SwahiliEvalCase(
        id="sw-general-soil",
        question="Ninawezaje kuboresha rutuba ya udongo shambani mwangu?",
        intent="general",
        expected_terms=("soil", "fertilizer"),
        notes="Soil fertility; open-ended general guidance.",
    ),
)


def list_cases(intent: str | None = None) -> list[SwahiliEvalCase]:
    """All eval cases, optionally filtered to one intent."""
    if intent is None:
        return list(SWAHILI_EVAL_SET)
    return [c for c in SWAHILI_EVAL_SET if c.intent == intent]


RUBRIC = (
    "A good Swahili answer: (1) is written in clear Kiswahili, (2) engages "
    "the expected domain concept (English term or its Swahili form both "
    "count), (3) is grounded - it does not invent specific figures or "
    "product names, and (4) is actionable for a smallholder. The grader "
    "measures (1) and (2) automatically; (3) and (4) are "
    "verifier's job."
)
