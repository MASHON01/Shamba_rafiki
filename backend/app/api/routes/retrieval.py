"""
Retrieval route - raw retrieval, no LLM.

    POST /retrieve   { query, top_k? }  ->  ranked chunks + scores

Exposes the retriever directly, bypassing the LLM. This is a
debugging / inspection tool: during Day 9 corpus QA you can see
exactly what chunks a query pulls and with what similarity, which is
the fastest way to diagnose bad answers ("was it the retrieval or the
generation?"). It's also useful in the demo to show the grounding.

If no index has been built yet, the retriever dependency is None and
this returns a clear, non-error response saying so rather than
failing - matching the tolerant-startup design.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_retriever
from app.api.middleware.schemas import RetrieveRequest
from app.core.responses import success_response

router = APIRouter(tags=["Retrieval"])


@router.post("/retrieve")
def retrieve(
    body: RetrieveRequest,
    retriever=Depends(get_retriever),
) -> dict:
    """
    Return the chunks most relevant to a query, with similarity
    scores. No LLM involved.
    """
    if retriever is None:
        return success_response(
            data={"results": [], "index_loaded": False},
            message=(
                "No vector index is loaded. Run the build_index script "
                "to enable retrieval."
            ),
        )

    results = retriever.retrieve(body.query, top_k=body.top_k)

    return success_response(
        data={
            "index_loaded": True,
            "count": len(results),
            "results": [
                {
                    "text": r.chunk.text,
                    "score": round(r.similarity_score, 4),
                    "crop": r.chunk.metadata.get("crop"),
                    "county": r.chunk.metadata.get("county"),
                    "source_filename": r.chunk.metadata.get("source_filename"),
                }
                for r in results
            ],
        },
        message=f"Retrieved {len(results)} chunk(s).",
    )