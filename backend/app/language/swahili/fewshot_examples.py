"""
Swahili few-shot exemplars.

One short, high-quality example answer per intent, in Kiswahili, showing
the model exactly the shape we want: grounded, cited with [Source N],
structured (cause -> steps -> cost for a diagnosis), and honest. A single
well-chosen exemplar is a strong, cheap steer on answer style and
language fidelity - the prompt-side complement to the glossary, and the
safe alternative to a fine-tune.

Deliberately one exemplar per intent to keep the prompt within the token
budget on a 4096-context model. The exemplars are illustrative style
guides, not facts to copy: they model *how* to answer, and explicitly
cite a source so the model learns to do the same.
"""

from __future__ import annotations

from app.config.constants import DEFAULT_INTENT

_FEWSHOT_HEADER = "Mfano wa jibu zuri (fuata mtindo huu, usinukuu maudhui):"

_EXEMPLARS: dict[str, str] = {
    "diagnosis": (
        "Swali: Mahindi yangu yana madoa ya kahawia na yananyauka, nifanyeje?\n"
        "Jibu:\n"
        "1. Chanzo kinachowezekana zaidi: ni ukungu (blight) [Source 1].\n"
        "2. Cha kufanya: (a) ondoa majani yaliyoathirika, (b) nyunyuzia dawa "
        "ya ukungu kama inavyoelekezwa kwenye marejeo, (c) epuka kumwagilia "
        "juu ya majani.\n"
        "3. Gharama: kama shamba limeathirika sana, linganisha gharama ya dawa "
        "na kupanda upya."
    ),
    "how_to": (
        "Swali: Nipande maharagwe lini na vipi?\n"
        "Jibu:\n"
        "1. Andaa udongo vizuri na uhakikishe una unyevu.\n"
        "2. Panda mbegu kwa nafasi inayoshauriwa kwenye marejeo [Source 1].\n"
        "3. Palilia mapema magugu yasipoote."
    ),
    "price": (
        "Swali: Bei ya maharagwe ikoje na niuze sasa?\n"
        "Jibu: Kwa mujibu wa marejeo, bei ni takriban [kiasi] kwa kilo "
        "[Source 1]. Kumbuka bei hubadilika na ni ya makadirio; linganisha na "
        "soko lako la karibu kabla ya kuuza."
    ),
    "general": (
        "Swali: Nizungushe mazao vipi ili kupunguza magonjwa?\n"
        "Jibu: Badilisha aina ya zao kila msimu (kwa mfano mahindi kisha "
        "maharagwe) ili kuvunja mzunguko wa magonjwa ya udongo [Source 1]."
    ),
}


def fewshot_block(intent: str) -> str:
    """
    The exemplar block for ``intent``, or the default-intent exemplar for
    an unknown intent. Always returns a non-empty block (there is a
    general exemplar), so callers can inject it unconditionally.
    """
    exemplar = _EXEMPLARS.get(intent) or _EXEMPLARS.get(DEFAULT_INTENT, "")
    if not exemplar:
        return ""
    return _FEWSHOT_HEADER + "\n" + exemplar
