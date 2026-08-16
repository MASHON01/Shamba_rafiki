"""
Orchestrator: the public entry point for the whole pipeline.

It owns the request lifecycle end to end: load history, choose a path,
run the stages via the dispatcher, verify the answer, record the turn,
and return a standardized response. It also handles the image path,
classifying a leaf photo and folding the result into retrieval.
"""

from __future__ import annotations

from app.core.responses import error_response, success_response
from app.models.language import LanguageCode
from app.models.request import ImageRequest, QueryRequest
from app.orchestration.dispatcher import Dispatcher, PreparedRequest
from app.orchestration.llm.base import BaseLLMClient
from app.orchestration.llm.llama_client import LLMError
from app.orchestration.memory import ConversationMemory
from app.orchestration.router import RequestRouter
from app.utils.logger import get_logger
from app.verification.verifier import Verifier

logger = get_logger("Orchestrator")

_UNAVAILABLE_EN = (
    "The assistant is temporarily unavailable. Please try again shortly, "
    "or ask a local agricultural extension officer."
)
_UNAVAILABLE_SW = (
    "Msaidizi hapatikani kwa sasa. Tafadhali jaribu tena baadaye, au uliza "
    "afisa wa ugani wa kilimo wa eneo lako."
)
_UNKNOWN = "unknown"


class Orchestrator:
    """Top-level request handler wiring memory, routing, and dispatch."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        dispatcher: Dispatcher | None = None,
        memory: ConversationMemory | None = None,
        router: RequestRouter | None = None,
        verifier: Verifier | None = None,
        classifier=None,
    ) -> None:
        self._memory = memory or ConversationMemory()
        self._router = router or RequestRouter()
        self._dispatcher = dispatcher or Dispatcher(llm_client=llm_client)
        self._verifier = verifier or Verifier()
        self._classifier = classifier

    def _get_classifier(self):
        if self._classifier is None:
            from app.vision import ImageClassifier

            self._classifier = ImageClassifier()
        return self._classifier

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def handle_query(self, request: QueryRequest) -> dict:
        logger.info(
            "orchestrator.query.received",
            request_id=str(request.request_id),
            session_id=request.session_id,
            language_hint=request.language,
            query_chars=len(request.query),
        )
        history = self._memory.get_history(request.session_id)
        plan = self._router.route_text()

        try:
            prepared = self._dispatcher.prepare(
                query=request.query, plan=plan, language_hint=request.language, history=history
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("orchestrator.prepare_failed", request_id=str(request.request_id))
            return error_response(
                "Something went wrong while answering. Please try again.",
                code="ORCHESTRATION_ERROR",
                details={"reason": str(exc)},
            )

        try:
            generation = self._dispatcher._llm.generate(prepared.prompt)
        except LLMError as exc:
            return self._degraded_response(request, prepared, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("orchestrator.unexpected_error", request_id=str(request.request_id))
            return error_response(
                "Something went wrong while answering. Please try again.",
                code="ORCHESTRATION_ERROR",
                details={"reason": str(exc)},
            )

        verified = self._verifier.verify(
            generation.text, prepared.context.sources, language=prepared.language
        )
        answer = verified.text
        self._memory.add_turn(request.session_id, request.query, answer)
        return self._success(request, prepared, generation, verified)

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    def handle_image_query(self, request: ImageRequest) -> dict:
        logger.info(
            "orchestrator.image.received",
            request_id=str(request.request_id),
            session_id=request.session_id,
            language_hint=request.language,
            has_text=bool(request.query),
        )
        classifier = self._get_classifier()
        if not classifier.is_available():
            if request.query:
                return self.handle_query(
                    QueryRequest(
                        query=request.query, language=request.language, session_id=request.session_id
                    )
                )
            return error_response(
                "Image classification isn't available on this device yet. "
                "Please describe the problem in words instead.",
                code="CLASSIFIER_UNAVAILABLE",
                details={"model_path": str(classifier.model_path)},
            )

        try:
            vision = classifier.predict(request.image_path)
        except FileNotFoundError as exc:
            return error_response(
                "The uploaded image could not be found.",
                code="IMAGE_NOT_FOUND",
                details={"reason": str(exc)},
            )
        except ValueError as exc:
            return error_response(
                "The uploaded file could not be read as an image.",
                code="INVALID_IMAGE",
                details={"reason": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("orchestrator.image.classify_failed")
            if request.query:
                return self.handle_query(
                    QueryRequest(
                        query=request.query, language=request.language, session_id=request.session_id
                    )
                )
            return error_response(
                "Something went wrong analyzing the image. Please try again.",
                code="CLASSIFICATION_ERROR",
                details={"reason": str(exc)},
            )

        is_sw = str(request.language).lower().startswith("sw")
        hint = self._image_retrieval_hint(vision)
        note = self._image_prompt_note(vision, is_sw)
        question = (request.query or "").strip() or self._synth_question(vision, is_sw)

        history = self._memory.get_history(request.session_id)
        plan = self._router.route_image()

        try:
            prepared = self._dispatcher.prepare(
                query=question,
                plan=plan,
                language_hint=request.language,
                history=history,
                retrieval_hint=hint,
                question_note=note,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("orchestrator.image.prepare_failed")
            return error_response(
                "Something went wrong while answering. Please try again.",
                code="ORCHESTRATION_ERROR",
                details={"reason": str(exc)},
            )

        try:
            generation = self._dispatcher._llm.generate(prepared.prompt)
        except LLMError as exc:
            response = self._degraded_response(request, prepared, exc)
            details = response.get("error", {}).get("details")
            if isinstance(details, dict):
                details["classification"] = self._classification_block(vision)
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("orchestrator.image.unexpected_error")
            return error_response(
                "Something went wrong while answering. Please try again.",
                code="ORCHESTRATION_ERROR",
                details={"reason": str(exc)},
            )

        verified = self._verifier.verify(
            generation.text, prepared.context.sources, language=prepared.language
        )
        answer = verified.text
        self._memory.add_turn(request.session_id, question, answer)
        return self._success(request, prepared, generation, verified, vision=vision)

    # ------------------------------------------------------------------
    # Shared response building
    # ------------------------------------------------------------------

    def _success(self, request, prepared, generation, verified, vision=None) -> dict:
        data = {
            "answer": verified.text,
            "language": prepared.analysis.detection.language.value,
            "intent": prepared.analysis.intent,
            "sources": self._format_sources(prepared.context.sources),
            "grounded": prepared.context.has_context,
            "confidence": verified.confidence_level.value,
            "verification_action": verified.action.value,
        }
        if vision is not None:
            data["classification"] = self._classification_block(vision)
        metadata = {
            "request_id": str(request.request_id),
            "session_id": request.session_id,
            "translated": prepared.analysis.translated,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "latency_ms": generation.latency_ms,
            "confidence_score": verified.confidence_score,
            "verification_flags": verified.report.flags,
        }
        return success_response(data=data, message="Answer generated.", metadata=metadata)

    def _degraded_response(self, request, prepared: PreparedRequest, exc: Exception) -> dict:
        sources = self._format_sources(prepared.context.sources)
        notice = _UNAVAILABLE_SW if prepared.language == LanguageCode.SWAHILI else _UNAVAILABLE_EN
        logger.info(
            "orchestrator.degraded", request_id=str(request.request_id), sources=len(sources)
        )
        return error_response(
            notice,
            code="LLM_UNAVAILABLE",
            details={
                "reason": str(exc),
                "retrieval_only": bool(sources),
                "sources": sources,
                "language": prepared.analysis.detection.language.value,
            },
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        client = self._dispatcher._llm
        llm_ok = client.health()
        data = {"llm_available": llm_ok}
        try:
            classifier = self._get_classifier()
            data["classifier"] = {
                "available": classifier.is_available(),
                "model_path": str(classifier.model_path),
            }
        except Exception as exc:  # noqa: BLE001
            data["classifier"] = {"available": False, "error": str(exc)}
        return success_response(data=data, message="ok" if llm_ok else "llm unavailable")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_retrieval_hint(vision) -> str:
        label = vision.best_label
        if vision.is_confident:
            return label.retrieval_hint()
        return "" if label.crop == _UNKNOWN else label.crop

    @staticmethod
    def _image_prompt_note(vision, is_sw: bool) -> str:
        label = vision.best_label
        band = vision.result.confidence_level.value
        crop = None if label.crop == _UNKNOWN else label.crop
        condition = None if label.condition == _UNKNOWN else label.condition
        subject = " - ".join(x for x in (crop, condition) if x)
        if is_sw:
            subject = subject or "hali isiyojulikana"
            return f"(Uchambuzi wa picha: yumkini {subject}; uhakika: {band}.)"
        subject = subject or "an unclear condition"
        return f"(Image analysis of the plant photo: likely {subject}; confidence: {band}.)"

    @staticmethod
    def _synth_question(vision, is_sw: bool) -> str:
        label = vision.best_label
        crop = None if label.crop == _UNKNOWN else label.crop
        if label.is_healthy:
            subject = crop or ("mmea" if is_sw else "the plant")
            if is_sw:
                return f"{subject} wangu unaonekana mzima. Niutunzeje na kuzuia magonjwa?"
            return f"My {subject} looks healthy. How do I keep it that way and prevent disease?"
        condition = None if label.condition == _UNKNOWN else label.condition
        subject = " ".join(x for x in (crop, condition) if x)
        if is_sw:
            subject = subject or "tatizo hili"
            return f"Mmea wangu wa {subject} unaonekana na tatizo. Nifanye nini kutibu na kudhibiti?"
        subject = subject or "this problem"
        return f"My plant appears to have {subject}. What should I do to treat and manage it?"

    @staticmethod
    def _classification_block(vision) -> dict:
        result = vision.result
        best = result.best_prediction
        return {
            "crop": best.crop,
            "condition": best.condition,
            "label": best.label,
            "confidence": round(best.confidence, 3),
            "confidence_level": result.confidence_level.value,
            "model": result.model_name,
            "inference_time_ms": result.inference_time_ms,
            "alternatives": [
                {"label": p.label, "crop": p.crop, "condition": p.condition,
                 "confidence": round(p.confidence, 3)}
                for p in result.predictions[1:]
            ],
        }

    @staticmethod
    def _format_sources(sources) -> list[dict]:
        formatted = []
        for i, result in enumerate(sources, start=1):
            meta = result.chunk.metadata
            formatted.append(
                {
                    "index": i,
                    "crop": meta.get("crop"),
                    "county": meta.get("county"),
                    "source_filename": meta.get("source_filename"),
                    "score": round(result.similarity_score, 3),
                }
            )
        return formatted
