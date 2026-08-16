"""
Model runtime: the GGUF model and the llama-server process it runs internally.,. delivered a ``LlamaClient`` that speaks HTTP to
a *running* llama-server. This package owns everything *underneath* that
seam - the model file and the server process the client talks to:

    registry candidate GGUFs + metadata (params, quant, size, RAM)
    downloader pinned Hugging Face download + checksum verification
    gguf_validator magic bytes / quantization / size sanity checks
    server_manager build llama-server args; start/stop the process
    readiness poll until the model is resident and answering
    model_info metadata surfaced to /health and the benchmark reports

The client is unchanged: it still just sends prompts to whatever server
this package brings up. Nothing here imports the client except the
readiness probe, which reuses it so there is one HTTP path, not two.
"""

from __future__ import annotations

from app.model_runtime.gguf_validator import (
    GGUFInfo,
    GGUFValidationError,
    validate_gguf,
)
from app.model_runtime.model_info import build_model_info
from app.model_runtime.readiness import ReadinessResult, wait_for_ready
from app.model_runtime.registry import (
    ModelSpec,
    default_model,
    get_model,
    list_models,
    resolved_model_path,
)

__all__ = [
    "ModelSpec",
    "get_model",
    "list_models",
    "default_model",
    "resolved_model_path",
    "GGUFInfo",
    "GGUFValidationError",
    "validate_gguf",
    "ReadinessResult",
    "wait_for_ready",
    "build_model_info",
]
