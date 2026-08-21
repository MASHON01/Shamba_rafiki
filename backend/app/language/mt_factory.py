"""
Translator construction.

`build_translator()` returns a `Translator` wired with the offline NLLB
engine when it is both enabled in settings and actually present on disk;
otherwise it returns a `Translator` with no engine, which uses the
dictionary-gloss fallback. Either way the caller gets a working
`Translator` and never has to care which backend is behind it.

This is deliberately tolerant: a missing model, a missing CTranslate2
install, or a disabled flag all resolve to "no engine", never an error.
"""

from __future__ import annotations

from app.language.translator import Translator
from app.utils.logger import get_logger

logger = get_logger("TranslatorFactory")


def build_translator() -> Translator:
    """Build the app-wide Translator, attaching the NLLB engine if available."""
    from app.config.settings import settings

    if not settings.SWAHILI_MT_ENABLED:
        logger.info("translator.engine.disabled")
        return Translator(engine=None)

    try:
        from app.language.mt_engine import NllbTranslatorEngine

        engine = NllbTranslatorEngine(
            model_dir=settings.SWAHILI_MT_MODEL_DIR,
            beam_size=settings.SWAHILI_MT_BEAM_SIZE,
        )
        if engine.is_available():
            logger.info("translator.engine.ready", backend="nllb-ct2")
            return Translator(engine=engine)
        logger.info(
            "translator.engine.absent",
            hint="Run scripts/download_translator.sh to enable fluent Swahili MT.",
        )
    except Exception as exc:  # noqa: BLE001 - engine is optional
        logger.warning("translator.engine.build_failed", reason=str(exc))

    return Translator(engine=None)
