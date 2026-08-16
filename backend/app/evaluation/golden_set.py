"""
Golden question set.

The curated questions the accuracy harness scores against. English is
primary (the judges evaluate mainly in English); Swahili is secondary but
real. Each case carries the qualities a good answer must have: the domain
concepts it should engage, whether it should be grounded in the corpus,
and which intent it is.

Two categories live here:
  - "standard": questions the corpus is designed to answer well.
  - "hidden": plausible, in-domain questions we did NOT tailor the corpus
    for - a self-run stand-in for the hidden prompts the judges add. The
    point is to catch overfitting to our own submission questions: a
    system that only shines on the questions it was built around is
    fragile. (Out-of-corpus "must say unsure" questions are separate -
    see hallucination_probes.py.)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One graded question and the qualities a good answer has."""

    id: str
    question: str
    intent: str
    language: str = "en"
    expected_terms: tuple[str, ...] = field(default_factory=tuple)
    expected_points: tuple[str, ...] = field(default_factory=tuple)
    must_ground: bool = True
    category: str = "standard"  # "standard" | "hidden"
    notes: str = ""

    # --- English (primary) ----------------------------------------------------


_EN_STANDARD: tuple[GoldenCase, ...] = (
    GoldenCase(
        id="en-diag-maize-blight",
        question="My maize has brown spots on the leaves and is wilting. What is wrong and what should I do?",
        intent="diagnosis",
        expected_terms=("blight", "maize", "fungicide"),
        expected_points=("remove affected", "rotate"),
        notes="Core maize blight diagnosis.",
    ),
    GoldenCase(
        id="en-diag-tomato-blight",
        question="My tomato plants have a grey mould on the leaves. How do I treat it?",
        intent="diagnosis",
        expected_terms=("blight", "tomato", "fungicide"),
    ),
    GoldenCase(
        id="en-howto-beans-planting",
        question="When and how should I plant beans in Kiambu?",
        intent="how_to",
        expected_terms=("beans", "seed"),
        expected_points=("spacing", "weed"),
    ),
    GoldenCase(
        id="en-price-beans",
        question="What is the price of beans in Nakuru and should I sell now or wait?",
        intent="price",
        expected_terms=("beans",),
        expected_points=("prices change", "approximate"),
        notes="Must frame prices as approximate + changing.",
    ),
    GoldenCase(
        id="en-diag-cassava-mosaic",
        question="My cassava leaves have a yellow mosaic pattern. What disease is this?",
        intent="diagnosis",
        expected_terms=("mosaic", "cassava", "disease"),
    ),
    GoldenCase(
        id="en-general-rotation",
        question="How should I rotate my crops to reduce disease?",
        intent="general",
        expected_terms=("disease", "soil"),
        expected_points=("rotate", "season"),
    ),
)

# Hidden prompts: in-domain but not corpus-tailored. Still gradeable on
# concept coverage; watched for honest grounding rather than confident
# invention.
_EN_HIDDEN: tuple[GoldenCase, ...] = (
    GoldenCase(
        id="en-hidden-maize-armyworm",
        question="How do I control fall armyworm in my maize?",
        intent="diagnosis",
        expected_terms=("maize", "pest", "pesticide"),
        must_ground=False,
        category="hidden",
        notes="Pest we may not have tailored the corpus for.",
    ),
    GoldenCase(
        id="en-hidden-maize-spacing",
        question="What is the recommended spacing for planting maize in Uasin Gishu?",
        intent="how_to",
        expected_terms=("maize", "seed"),
        must_ground=False,
        category="hidden",
        notes="A specific agronomic detail the corpus may lack.",
    ),
)

# --- Swahili (secondary) --------------------------------------------------

_SW: tuple[GoldenCase, ...] = (
    GoldenCase(
        id="sw-diag-maize-blight",
        question="Mahindi yangu yana madoa ya kahawia na yananyauka, nifanyeje?",
        intent="diagnosis",
        language="sw",
        expected_terms=("blight", "maize", "fungicide"),
    ),
    GoldenCase(
        id="sw-diag-cassava-mosaic",
        question="Muhogo wangu una batobato kwenye majani, ni ugonjwa gani?",
        intent="diagnosis",
        language="sw",
        expected_terms=("mosaic", "cassava"),
    ),
    GoldenCase(
        id="sw-price-beans",
        question="Bei ya maharagwe ikoje Nakuru, niuze sasa au nisubiri?",
        intent="price",
        language="sw",
        expected_terms=("beans",),
    ),
)

GOLDEN_SET: tuple[GoldenCase, ...] = _EN_STANDARD + _EN_HIDDEN + _SW


def golden_cases(language: str | None = None, category: str | None = None) -> list[GoldenCase]:
    """Golden cases, optionally filtered by language and/or category."""
    cases = list(GOLDEN_SET)
    if language is not None:
        cases = [c for c in cases if c.language == language]
    if category is not None:
        cases = [c for c in cases if c.category == category]
    return cases
