"""
Response-shape guidance (structured output).

Reusable, per-intent instructions on *how the answer should be shaped* -
diagnosis as cause -> steps -> cost, price as figures -> recommendation ->
caveat, and so on. Kept separate from the persona/grounding rules so the
two can evolve independently: a template version composes a base prompt
(who the model is + grounding rules) with the shape guidance for the
intent.

Only the hardened v2 templates use these today; v1 keeps its original
inline phrasing so the baseline is unchanged. English and Swahili are
kept in lockstep so a Swahili answer is shaped the same as an English
one.
"""

from __future__ import annotations

# --- English -------------------------------------------------------------

OUTPUT_FORMAT_EN: dict[str, str] = {
    "diagnosis": (
        "Shape the answer as: (1) Most likely cause - name it and cite the "
        "source you took it from, like [Source 1]. (2) What to do - clear "
        "numbered steps a smallholder can follow in order. (3) Cost check - "
        "one line on whether treating is worth it versus replanting."
    ),
    "price": (
        "Shape the answer as: (1) The price or market figures, each with its "
        "[Source N]. (2) A short, plain recommendation. (3) One line noting "
        "that prices change and are approximate."
    ),
    "how_to": (
        "Shape the answer as clear numbered steps in the correct order, each "
        "short enough to act on. Add a one-line timing or cost note only if "
        "the reference material gives one."
    ),
    "general": (
        "Answer directly and briefly, citing the source you used like "
        "[Source 1] whenever you state a specific fact."
    ),
}

# --- Swahili --------------------------------------------------------------

OUTPUT_FORMAT_SW: dict[str, str] = {
    "diagnosis": (
        "Panga jibu hivi: (1) Chanzo kinachowezekana zaidi - kitaje na utaje "
        "chanzo ulichokitumia, kama [Source 1]. (2) Cha kufanya - hatua zenye "
        "namba ambazo mkulima mdogo anaweza kufuata kwa mpangilio. (3) Ukaguzi "
        "wa gharama - mstari mmoja kama matibabu yanafaa dhidi ya kupanda upya."
    ),
    "price": (
        "Panga jibu hivi: (1) Takwimu za bei au soko, kila moja na [Source N] "
        "yake. (2) Pendekezo fupi na wazi. (3) Mstari mmoja ukikumbusha kwamba "
        "bei hubadilika na ni za makadirio."
    ),
    "how_to": (
        "Panga jibu kama hatua zenye namba kwa mpangilio sahihi, kila moja "
        "fupi ya kutekelezeka. Ongeza dokezo moja la muda au gharama tu kama "
        "maelezo ya rejea yanalitoa."
    ),
    "general": (
        "Jibu moja kwa moja na kwa kifupi, ukitaja chanzo ulichokitumia kama "
        "[Source 1] wakati wowote unaposema jambo mahususi."
    ),
}
