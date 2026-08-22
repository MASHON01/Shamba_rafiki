#!/usr/bin/env bash
#
# ADTC submission model downloader.
#
# Downloads the GGUF weight file to model/, matching `_runtime.model_path`
# in metadata.json. Idempotent (skips if already present) and requires no
# credentials — the weights are hosted publicly on the Hugging Face Hub.
#
#     bash download_model.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$REPO_ROOT/model"
MODEL_FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# Public, no-auth GGUF (bartowski's Llama 3.2 1B Instruct quantizations).
URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/${MODEL_FILE}?download=true"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_PATH" ]; then
  echo "Model already present, skipping download: $MODEL_PATH"
  exit 0
fi

echo "Downloading $MODEL_FILE (~2.0 GB) to $MODEL_PATH ..."
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "$MODEL_PATH" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$MODEL_PATH" "$URL"
else
  echo "ERROR: neither curl nor wget is available." >&2
  exit 1
fi

echo "Done: $MODEL_PATH"
