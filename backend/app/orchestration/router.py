"""
Request router.

Decides the *path* a request should take through the pipeline -
purely a decision, no execution (that's the dispatcher's job).
Separating the two keeps each trivially testable: the router is
tested on the routes it chooses, the dispatcher on the stages it runs.

For Phase 1 the routing is simple - every text query follows the
retrieval-augmented path. The value of having this as its own module
is Phase 3: when image input arrives, the router is the one place
that decides "image present -> classify first, then retrieve" without
the orchestrator or dispatcher changing shape.

A RoutePlan says which stages to run and carries the resolved intent
(from language analysis) so the dispatcher knows which prompt to
build.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    """The pipeline paths a request can take."""

    RAG_TEXT = "rag_text"          # text -> language -> retrieve -> prompt -> llm
    IMAGE_RAG = "image_rag"        # image -> classify -> retrieve -> ... (Phase 3)


@dataclass(slots=True)
class RoutePlan:
    """
    What the dispatcher should execute for a request.
    """

    route: Route
    use_retrieval: bool
    use_classifier: bool


class RequestRouter:
    """
    Chooses a RoutePlan for a request.
    """

    def route_text(self) -> RoutePlan:
        """Path for a text-only query (the Phase 1 path)."""
        return RoutePlan(
            route=Route.RAG_TEXT,
            use_retrieval=True,
            use_classifier=False,
        )

    def route_image(self) -> RoutePlan:
        """
        Path for an image (+optional text) request. Reserved for
        Phase 3 - the classifier runs first, then retrieval folds in
        its label. Declared now so the shape is stable; the dispatcher
        will gain the classifier call when Phase 3 lands.
        """
        return RoutePlan(
            route=Route.IMAGE_RAG,
            use_retrieval=True,
            use_classifier=True,
        )