"""
Generation configuration and sampling control.

One tested place that decides *how* the model generates - temperature,
top-p, top-k, token budget, stop sequences, seed, repeat penalty - so
sampling behaviour is deliberate and reviewable, not scattered across
call sites as loose keyword arguments.

Two design points matter for this product:

  - Low temperature by default. A farmer needs a consistent, grounded
    answer, not a creative one. Advice that changes each time it is
    asked is worse than useless, so the defaults sit low and the
    per-intent presets (see presets.py) only nudge from there.

  - Guardrails, not just defaults. ``max_tokens`` is clamped to a hard
    cap (LLM_MAX_TOKENS_CAP) no matter what a preset, per-request
    override, or stray env value asks for. Runaway generation is the
    main controllable risk to latency and RAM on the 8 GB target, so no
    configuration is allowed to breach the wall.

A ``deterministic`` variant pins the seed and drops to greedy decoding
so evaluation runs score the prompt and model, not sampling
noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.config.constants import (
    LLM_DETERMINISTIC_SEED,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_CAP,
    LLM_REPEAT_PENALTY,
    LLM_STOP_SEQUENCES,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
)


def _default_stop() -> list[str]:
    # A copy, so no config can mutate the shared constant in place.
    return list(LLM_STOP_SEQUENCES)


@dataclass(slots=True)
class GenerationConfig:
    """
    Sampling parameters for one generation.

    All fields have grounded defaults from constants, so an empty
    ``GenerationConfig`` is the safe, low-temperature house style.
    ``__post_init__`` enforces the guardrails, so an instance is always
    valid to hand to the model - callers never have to re-check.
    """

    temperature: float = LLM_TEMPERATURE
    top_p: float = LLM_TOP_P
    top_k: int = LLM_TOP_K
    max_tokens: int = LLM_MAX_TOKENS
    repeat_penalty: float = LLM_REPEAT_PENALTY
    seed: int | None = None
    stop: list[str] = field(default_factory=_default_stop)

    def __post_init__(self) -> None:
        # Clamp into valid, safe ranges rather than raising: a slightly
        # out-of-range knob should be corrected, not crash a farmer's
        # query. The one thing we protect hard is the token ceiling.
        self.temperature = _clamp(float(self.temperature), 0.0, 2.0)
        self.top_p = _clamp(float(self.top_p), 0.0, 1.0)
        self.top_k = max(0, int(self.top_k))
        self.repeat_penalty = _clamp(float(self.repeat_penalty), 0.0, 2.0)

        max_tokens = int(self.max_tokens)
        if max_tokens < 1:
            max_tokens = 1
            # The wall: never exceed the hard cap, whatever was asked.
        self.max_tokens = min(max_tokens, LLM_MAX_TOKENS_CAP)

        if self.seed is not None:
            self.seed = int(self.seed)

            # Normalize stop to a clean list of non-empty strings.
        self.stop = [s for s in (self.stop or []) if s]

        # ------------------------------------------------------------------
        # Derivations
        # ------------------------------------------------------------------

    def override(self, **changes) -> "GenerationConfig":
        """
        A new config with ``changes`` applied (guardrails re-run).

        Used for per-request overrides on top of a preset without
        mutating the shared preset instance.
        """
        return replace(self, **changes)

    def deterministic(self, seed: int = LLM_DETERMINISTIC_SEED) -> "GenerationConfig":
        """
        A reproducible variant: fixed seed + greedy (temperature 0).

        For evaluation runs, so a score reflects the prompt and model,
        not the luck of the sampler.
        """
        return replace(self, seed=seed, temperature=0.0)

    def to_payload(self) -> dict:
        """
        Sampling fields as a llama-server /completion payload fragment.

        Only the sampling keys - the client adds ``prompt`` and
        ``stream``. ``seed`` is omitted when None so llama-server keeps
        its own (random) default rather than being pinned.
        """
        payload: dict[str, object] = {
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "stop": list(self.stop),
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
