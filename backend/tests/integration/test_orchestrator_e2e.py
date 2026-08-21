"""Integration test: the full orchestrator pipeline.

Request -> Language -> Retrieval -> Prompt -> LLM -> Verification ->
Response, end to end, with the fake LLM + real retriever from
conftest. Confirms the stages actually connect: retrieved context
reaches the prompt, verification runs on the answer, memory threads
follow-ups, and failures degrade cleanly."""

from __future__ import annotations

import pytest

from app.models.request import QueryRequest
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.llama_client import LLMConnectionError

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_full_pipeline_grounded_answer(orchestrator, fake_llm):
    resp = orchestrator.handle_query(QueryRequest(
        query="How do I treat maize blight in Nakuru?",
        language="en", session_id="s1"))
    assert resp["success"]
    d = resp["data"]
    assert d["answer"]
    assert d["grounded"] is True
    assert d["intent"] in ("diagnosis", "how_to")
    assert len(d["sources"]) >= 1
    assert d["sources"][0]["crop"] == "maize"
    # verification ran
    assert "confidence" in d and "verification_action" in d


def test_retrieved_context_reaches_prompt(orchestrator, fake_llm):
    orchestrator.handle_query(QueryRequest(
        query="maize blight treatment", language="en"))
    # the fake LLM records the prompt it was handed
    assert fake_llm.last_prompt is not None
    assert "Farm Pal" in fake_llm.last_prompt.system_prompt
    assert "maize" in fake_llm.last_prompt.full_prompt.lower()


def test_memory_threads_followup(orchestrator, fake_llm):
    orchestrator.handle_query(QueryRequest(
        query="How do I treat maize blight?", language="en", session_id="farmer"))
    orchestrator.handle_query(QueryRequest(
        query="And how much does it cost?", language="en", session_id="farmer"))
    # second prompt should carry the first Q as history
    assert "maize blight" in fake_llm.last_prompt.full_prompt.lower()


def test_sessions_are_isolated(orchestrator, fake_llm):
    orchestrator.handle_query(QueryRequest(
        query="tomato blight question", language="en", session_id="A"))
    orchestrator.handle_query(QueryRequest(
        query="fresh unrelated question", language="en", session_id="B"))
    # session B's prompt must not contain session A's history
    assert "tomato blight question" not in fake_llm.last_prompt.full_prompt


def test_swahili_query_end_to_end(orchestrator):
    resp = orchestrator.handle_query(QueryRequest(
        query="Mahindi yangu yana ukungu, nifanye nini?",
        language="sw", session_id="sw1"))
    assert resp["success"]
    assert resp["data"]["language"] == "sw"


def test_llm_failure_degrades_cleanly(retriever):
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    class DeadLLM(BaseLLMClient):
        def generate(self, prompt):
            raise LLMConnectionError("llama-server down")
        def health(self):
            return False

    orch = Orchestrator(
        llm_client=DeadLLM(),
        dispatcher=Dispatcher(llm_client=DeadLLM(), retriever=retriever))
    resp = orch.handle_query(QueryRequest(query="test", language="en"))
    assert not resp["success"]
    assert resp["error"]["code"] == "LLM_UNAVAILABLE"


def test_degraded_no_retriever_still_answers(fake_llm):
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    orch = Orchestrator(
        llm_client=fake_llm,
        dispatcher=Dispatcher(llm_client=fake_llm, retriever=None))
    resp = orch.handle_query(QueryRequest(query="hello", language="en"))
    assert resp["success"]
    assert resp["data"]["grounded"] is False