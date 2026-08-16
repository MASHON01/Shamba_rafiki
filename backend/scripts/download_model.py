#!/usr/bin/env python3
"""
Download and verify a GGUF model from the registry.

    python -m scripts.download_model # the default model
    python -m scripts.download_model --list # show candidates
    python -m scripts.download_model --model qwen2.5-3b-instruct-q4_k_m
    python -m scripts.download_model --force # re-download
    python -m scripts.download_model --print-command # llama-server cmd

One command fetches the pinned file from Hugging Face, checks its GGUF
magic bytes / quantization / size, verifies (or records) its SHA256, and
tells you the exact llama-server command to bring it up. This is the
"managed model layer" deliverable: reproducible fetch + verify, and a
documented way to serve it.

The download needs network + huggingface_hub. Verification and the
command printout are offline.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.logging import configure_logging  # noqa: E402
from app.model_runtime.downloader import download_model  # noqa: E402
from app.model_runtime.registry import (  # noqa: E402
    default_model,
    get_model,
    list_models,
)
from app.model_runtime.server_manager import (  # noqa: E402
    ServerConfig,
    command_string,
)
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("DownloadModel")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download + verify a GGUF model from the registry."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Registry id to fetch (default: the configured MODEL_ID).",
    )
    parser.add_argument("--list", action="store_true", help="List candidate models and exit.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if a verified file already exists."
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip verification (not recommended)."
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Also print the llama-server command for this model.",
    )
    args = parser.parse_args()

    configure_logging()

    if args.list:
        _print_catalogue()
        return 0

    spec = get_model(args.model) if args.model else default_model

    print(f"\nModel: {spec.display_name} [{spec.id}]")
    print(f"Source: {spec.hf_url}")
    print(
        f"Quant: {spec.quantization} ~{spec.file_size_mb} MB on disk, "
        f"~{spec.est_ram_mb} MB RAM (estimate)\n"
    )

    try:
        path = download_model(spec, force=args.force, verify=not args.no_verify)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error.
        print(f"[!!] Download/verification failed: {exc}")
        return 1

    print(f"[ok] Ready: {path}")

    if args.print_command:
        config = ServerConfig.from_settings(spec=spec, model_path=path)
        print("\nStart llama-server with:\n")
        print(f" {command_string(config)}\n")

    return 0


def _print_catalogue() -> None:
    # Listing must work even if MODEL_ID is misconfigured.
    active = None
    with contextlib.suppress(Exception):
        active = default_model.id

    print("\nRegistered models")
    print("=" * 60)
    for spec in list_models:
        marker = " *" if spec.id == active else " "
        print(f"{marker} {spec.id}")
        print(
            f" {spec.display_name} - {spec.quantization}, "
            f"~{spec.file_size_mb} MB, ~{spec.est_ram_mb} MB RAM"
        )
    print("\n * = active (settings.MODEL_ID)\n")


if __name__ == "__main__":
    raise SystemExit(main)
