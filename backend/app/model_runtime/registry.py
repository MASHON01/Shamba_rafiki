"""
Candidate GGUF model registry.

A small, explicit catalogue of the models we are willing to run, each
pinned to a specific Hugging Face repo, revision, and file. This is the
single source of truth for "which models exist and what do they cost",
consumed by the downloader (what to fetch), the GGUF validator (what to
expect), the server manager (which file to serve), and /health + reports
(what is running).

Selection principle (from the build plan): on an 8 GB machine with a
7 GB hard ceiling, RAM headroom beats raw speed. A 3B model at Q4_K_M
leaves ~50% headroom - that headroom is the design target, not a side
effect, because it is what wins the Efficiency score and the Budget
Profile bonus. So the default is a 3B Q4_K_M, with a 1B fallback if a
weaker machine ever forces it.

    file_size_mb / est_ram_mb are ESTIMATES, indicative only. The real
    RAM and latency numbers come from, run on the 8 GB target
    machine with the real GGUF. Nothing here counts toward the rubric.

Revisions are pinned to a branch name by default ("main"). Before the
final submission build, pin each `revision` to a concrete commit SHA so
a fresh download is bit-for-bit reproducible; record the resulting
SHA256 in `sha256` so future downloads are verified, not trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import paths
from app.config.settings import settings
from app.core.exceptions import RegistryError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    Everything needed to fetch, verify, serve, and describe one GGUF.
    """

    id: str
    display_name: str
    repo_id: str
    filename: str
    revision: str = "main"

    # Descriptive metadata (surfaced to /health and reports).
    params_b: float = 0.0
    quantization: str = "Q4_K_M"
    context_length: int = 4096

    # Budget estimates - indicative only, replaced by real numbers in
    # on the target machine.
    file_size_mb: int = 0
    est_ram_mb: int = 0

    # Integrity. Filled once known so downloads are verified, not trusted.
    sha256: str | None = None

    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def hf_url(self) -> str:
        """Human-readable pointer to the exact file on Hugging Face."""
        return f"https://huggingface.co/{self.repo_id}/blob/" f"{self.revision}/{self.filename}"

        # ---------------------------------------------------------------------------
        # The catalogue
        # ---------------------------------------------------------------------------
        #
        # GGUF sources are bartowski's quantizations - the correct, existing
        # source for Llama 3.2 and Qwen2.5 GGUFs (the `hugging-quants` org does
        # not publish Gemma/Qwen GGUFs, and returns a misleading 401 for repos
        # that do not exist). Filenames are case-sensitive: bartowski uses an
        # uppercase `Q4_K_M.gguf` suffix.


_MODELS: dict[str, ModelSpec] = {
    "llama-3.2-3b-instruct-q4_k_m": ModelSpec(
        id="llama-3.2-3b-instruct-q4_k_m",
        display_name="Llama 3.2 3B Instruct (Q4_K_M)",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        revision="main",
        params_b=3.2,
        quantization="Q4_K_M",
        context_length=131072,
        file_size_mb=2019,
        est_ram_mb=2600,
        notes=(
            "Primary/default. Conservative 3B leaves large headroom under "
            "the 7 GB ceiling. Strong English instruction following; "
            "Swahili proven separately internally."
        ),
        tags=("default", "primary"),
    ),
    "qwen2.5-3b-instruct-q4_k_m": ModelSpec(
        id="qwen2.5-3b-instruct-q4_k_m",
        display_name="Qwen2.5 3B Instruct (Q4_K_M)",
        repo_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        revision="main",
        params_b=3.1,
        quantization="Q4_K_M",
        context_length=32768,
        file_size_mb=1930,
        est_ram_mb=2500,
        notes=(
            "Alternate candidate for bake-off. Compare on "
            "RAM + latency + answer accuracy against the Llama default."
        ),
        tags=("alternate",),
    ),
    "llama-3.2-1b-instruct-q4_k_m": ModelSpec(
        id="llama-3.2-1b-instruct-q4_k_m",
        display_name="Llama 3.2 1B Instruct (Q4_K_M)",
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        revision="main",
        params_b=1.2,
        quantization="Q4_K_M",
        context_length=131072,
        file_size_mb=808,
        est_ram_mb=1200,
        notes=(
            "Low-RAM fallback if a weaker-than-target machine ever forces "
            "it. Faster and lighter, but weaker reasoning - a safety net, "
            "not the intended ship model."
        ),
        tags=("fallback", "low-ram"),
    ),
}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def list_models() -> list[ModelSpec]:
    """All registered candidate models, in catalogue order."""
    return list(_MODELS.values())


def get_model(model_id: str) -> ModelSpec:
    """
    Return the spec for ``model_id``.

    Raises
    ------
    app.core.exceptions.RegistryError
        If no model with that id is registered - listing the ids that
        are, so a typo in MODEL_ID is easy to fix.
    """
    try:
        return _MODELS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(_MODELS)) or "(none)"
        raise RegistryError(f"Unknown model id {model_id!r}. Registered models: {known}.") from exc


def default_model() -> ModelSpec:
    """
    The active model, chosen by ``settings.MODEL_ID``.

    Falls back to raising a clear RegistryError (via get_model) if the
    configured id is not registered, rather than silently guessing.
    """
    return get_model(settings.MODEL_ID)


def resolved_model_path(spec: ModelSpec, models_dir: Path | None = None) -> Path:
    """
    Canonical on-disk location of a spec's GGUF: ``models/<filename>``.

    This is the single convention for where a downloaded model lives, so
    the downloader, server manager, validator, and /health all agree on
    the path without threading it through settings. ``models_dir``
    defaults to the repo-root ``models/`` directory.
    """
    base = models_dir if models_dir is not None else paths.MODELS_DIR
    return base / spec.filename
