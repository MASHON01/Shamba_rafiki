"""
Unit tests for generation configuration and sampling control.

Covers the config guardrails, the deterministic variant, per-intent
preset selection, the client's payload forwarding, and the dispatcher
wiring (preset by intent + per-request override) - all offline.
"""

from __future__ import annotations

import types

import pytest
from app.config.constants import (
    DEFAULT_INTENT,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_CAP,
    LLM_TEMPERATURE,
)
from app.orchestration.llm.generation_config import GenerationConfig
from app.orchestration.llm.llama_client import LlamaClient
from app.orchestration.llm.presets import PRESETS, for_intent
from app.orchestration.router import RequestRouter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# GenerationConfig defaults + guardrails
# ---------------------------------------------------------------------------


def test_defaults_are_house_style():
    cfg = GenerationConfig
    assert cfg.temperature == LLM_TEMPERATURE
    assert cfg.max_tokens == LLM_MAX_TOKENS
    assert cfg.seed is None
    assert cfg.stop  # non-empty default stop list


def test_max_tokens_hard_capped():
    cfg = GenerationConfig(max_tokens=10_000)
    assert cfg.max_tokens == LLM_MAX_TOKENS_CAP


def test_max_tokens_floored_to_one():
    assert GenerationConfig(max_tokens=0).max_tokens == 1
    assert GenerationConfig(max_tokens=-5).max_tokens == 1


def test_temperature_and_top_p_clamped():
    hot = GenerationConfig(temperature=9.0, top_p=5.0)
    assert hot.temperature == 2.0
    assert hot.top_p == 1.0
    cold = GenerationConfig(temperature=-1.0, top_p=-1.0)
    assert cold.temperature == 0.0
    assert cold.top_p == 0.0


def test_stop_normalized_drops_empties():
    cfg = GenerationConfig(stop=["", "\nQuestion:", ""])
    assert cfg.stop == ["\nQuestion:"]


def test_default_stop_is_a_copy_not_shared():
    a = GenerationConfig
    a.stop.append("XXX")
    b = GenerationConfig
    assert "XXX" not in b.stop

    # ---------------------------------------------------------------------------
    # Derivations: override / deterministic / to_payload
    # ---------------------------------------------------------------------------


def test_override_returns_new_and_reapplies_guardrails():
    base = GenerationConfig(temperature=0.2)
    changed = base.override(temperature=0.9, max_tokens=99_999)
    assert changed.temperature == 0.9
    assert changed.max_tokens == LLM_MAX_TOKENS_CAP  # guardrail re-run
    assert base.temperature == 0.2  # original untouched


def test_deterministic_pins_seed_and_greedy():
    cfg = GenerationConfig(temperature=0.7).deterministic
    assert cfg.temperature == 0.0
    assert cfg.seed == 0


def test_to_payload_shape_and_seed_omission():
    cfg = GenerationConfig(top_k=13, repeat_penalty=1.2)
    payload = cfg.to_payload
    assert payload["n_predict"] == cfg.max_tokens
    assert payload["top_k"] == 13
    assert payload["repeat_penalty"] == 1.2
    assert "seed" not in payload  # None seed -> omitted

    seeded = cfg.override(seed=42).to_payload
    assert seeded["seed"] == 42

    # ---------------------------------------------------------------------------
    # Per-intent presets
    # ---------------------------------------------------------------------------


def test_factual_intents_run_colder_than_explanatory():
    assert for_intent("diagnosis").temperature == 0.20
    assert for_intent("price").temperature == 0.20
    assert for_intent("how_to").temperature == 0.35
    assert for_intent("diagnosis").temperature < for_intent("how_to").temperature


def test_unknown_intent_falls_back_to_default():
    assert for_intent("nonsense").temperature == PRESETS[DEFAULT_INTENT].temperature
    assert for_intent(None).temperature == PRESETS[DEFAULT_INTENT].temperature

    # ---------------------------------------------------------------------------
    # LlamaClient forwards config into the /completion payload
    # ---------------------------------------------------------------------------


def _fake_requests(captured: dict):
    """A stand-in `requests` module that captures the POSTed JSON."""

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"content": "ok", "tokens_evaluated": 3, "tokens_predicted": 2}

    def post(url, json, timeout):
        captured.update(json)
        return _Resp

    exceptions = types.SimpleNamespace(
        Timeout=type("Timeout", (Exception,), {}),
        ConnectionError=type("ConnectionError", (Exception,), {}),
        RequestException=type("RequestException", (Exception,), {}),
    )
    return types.SimpleNamespace(post=post, exceptions=exceptions)


class _Prompt:
    full_prompt = "SYSTEM\n\nQuestion: hi"


def test_client_forwards_config_payload(monkeypatch):
    captured: dict = {}
    client = LlamaClient(server_url="http://x")
    monkeypatch.setattr(client, "_load_requests", lambda: _fake_requests(captured))

    cfg = GenerationConfig(top_k=7, repeat_penalty=1.3, max_tokens=50).override(seed=99)
    client.generate(_Prompt, cfg)

    assert captured["top_k"] == 7
    assert captured["repeat_penalty"] == 1.3
    assert captured["n_predict"] == 50
    assert captured["seed"] == 99
    assert captured["stream"] is False


def test_client_without_config_uses_instance_defaults(monkeypatch):
    captured: dict = {}
    client = LlamaClient(server_url="http://x", max_tokens=123, temperature=0.5)
    monkeypatch.setattr(client, "_load_requests", lambda: _fake_requests(captured))

    client.generate(_Prompt)  # no config

    assert captured["n_predict"] == 123
    assert captured["temperature"] == 0.5
    assert "top_k" not in captured  # legacy path stays minimal

    # ---------------------------------------------------------------------------
    # Dispatcher wiring: preset by intent + per-request override
    # ---------------------------------------------------------------------------


def _dispatcher(fake_llm):
    from app.orchestration.dispatcher import Dispatcher

    # Real analyzer (cheap, offline), no retriever needed for this check.
    return Dispatcher(llm_client=fake_llm, retriever=None)


def test_dispatch_selects_preset_by_intent(fake_llm):
    dispatcher = _dispatcher(fake_llm)
    plan = RequestRouter.route_text
    dispatcher.dispatch(
        query="My maize has yellow spots and is wilting - what disease is this?",
        plan=plan,
        language_hint="en",
    )
    # Diagnosis intent -> cold factual preset.
    assert fake_llm.last_config is not None
    assert fake_llm.last_config.temperature == 0.20


def test_dispatch_per_request_override_wins(fake_llm):
    dispatcher = _dispatcher(fake_llm)
    plan = RequestRouter.route_text
    override = GenerationConfig(temperature=0.9, max_tokens=5)
    dispatcher.dispatch(
        query="My maize has yellow spots and is wilting?",
        plan=plan,
        language_hint="en",
        generation_config=override,
    )
    assert fake_llm.last_config.temperature == 0.9
    assert fake_llm.last_config.max_tokens == 5
