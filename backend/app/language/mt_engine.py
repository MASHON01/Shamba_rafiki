"""
Offline neural machine-translation engine (NLLB via CTranslate2).

This is the fluent-MT backend that plugs into the `TranslatorEngine`
seam in translator.py. It runs NLLB-200-distilled-600M through
CTranslate2 on the CPU, so translation stays fully offline and does
not keep a PyTorch graph resident at serve time (CTranslate2 is a
small C++ runtime; only the tokenizer comes from transformers).

Why it exists: a 1B chat model is weak at composing fluent Kiswahili
while simultaneously reasoning over English reference text. Splitting
the job - retrieve and answer in English (its strong language), then
translate the finished answer to Kiswahili - gives markedly cleaner
Swahili. The same engine also translates a Swahili question into
English up front so retrieval matches the English corpus.

Everything here is lazy and defensive. If CTranslate2, the tokenizer,
or the converted model directory is missing, `is_available()` returns
False and the caller silently falls back to the dictionary gloss +
Swahili-prompt path. Nothing about startup or the English path depends
on this being present.

Build the model once, offline, with scripts/download_translator.sh.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.config.paths import ROOT_DIR
from app.models.language import LanguageCode
from app.utils.logger import get_logger

logger = get_logger("NllbTranslator")

# NLLB uses FLORES-200 language codes, not ISO-639-1.
_NLLB_CODE: dict[LanguageCode, str] = {
    LanguageCode.ENGLISH: "eng_Latn",
    LanguageCode.SWAHILI: "swh_Latn",
}


class NllbTranslatorEngine:
    """
    CTranslate2-backed NLLB translator satisfying the TranslatorEngine
    protocol: `translate(text, source, target) -> str`.

    Model and tokenizer are loaded on first use (not at construction),
    so importing this module and building the object are both cheap and
    never raise. A failed load is remembered so we don't retry on every
    call.
    """

    def __init__(
        self,
        model_dir: str | Path,
        beam_size: int = 2,
        max_input_length: int = 512,
    ) -> None:
        self._model_dir = Path(model_dir)
        if not self._model_dir.is_absolute():
            self._model_dir = ROOT_DIR / self._model_dir
        self._beam_size = beam_size
        self._max_input_length = max_input_length

        self._lock = threading.Lock()
        self._loaded = False
        self._load_failed = False
        self._translator = None  # ctranslate2.Translator
        self._tokenizer = None  # transformers tokenizer

    # -- availability ---------------------------------------------------

    def is_available(self) -> bool:
        """True if the model can be (or already is) loaded."""
        if self._loaded:
            return True
        if self._load_failed:
            return False
        if not self._model_dir.is_dir():
            return False
        self._ensure_loaded()
        return self._loaded

    def _ensure_loaded(self) -> None:
        if self._loaded or self._load_failed:
            return
        with self._lock:
            if self._loaded or self._load_failed:
                return
            try:
                import ctranslate2  # noqa: PLC0415 - lazy, optional dep
                from transformers import AutoTokenizer  # noqa: PLC0415

                self._translator = ctranslate2.Translator(
                    str(self._model_dir), device="cpu"
                )
                self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
                self._loaded = True
                logger.info("translator.nllb.loaded", model_dir=str(self._model_dir))
            except Exception as exc:  # noqa: BLE001 - optional capability
                self._load_failed = True
                logger.warning(
                    "translator.nllb.unavailable",
                    reason=str(exc),
                    hint="Run scripts/download_translator.sh to enable fluent MT.",
                )

    # -- translation ----------------------------------------------------

    def translate(
        self,
        text: str,
        source: LanguageCode = LanguageCode.SWAHILI,
        target: LanguageCode = LanguageCode.ENGLISH,
    ) -> str:
        """
        Translate `text` from `source` to `target`. Returns the input
        unchanged if the model is unavailable or the text is empty, so
        this never becomes a hard failure on the request path.
        """
        text = (text or "").strip()
        if not text:
            return text

        src = _NLLB_CODE.get(source)
        tgt = _NLLB_CODE.get(target)
        if src is None or tgt is None:
            return text

        if not self.is_available():
            return text

        try:
            return self._run(text, src, tgt)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash a query
            logger.warning("translator.nllb.translate_failed", reason=str(exc))
            return text

    def _run(self, text: str, src: str, tgt: str) -> str:
        tokenizer = self._tokenizer
        tokenizer.src_lang = src
        encoded = tokenizer.encode(text)[: self._max_input_length]
        source_tokens = tokenizer.convert_ids_to_tokens(encoded)

        results = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[[tgt]],
            beam_size=self._beam_size,
            max_input_length=self._max_input_length,
        )
        target_tokens = list(results[0].hypotheses[0])

        # Drop the forced target-language token that prefixes the output.
        if target_tokens and target_tokens[0] == tgt:
            target_tokens = target_tokens[1:]

        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        return tokenizer.decode(target_ids, skip_special_tokens=True).strip()
