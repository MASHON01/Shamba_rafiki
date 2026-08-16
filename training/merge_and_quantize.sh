#!/usr/bin/env bash
#
# Merge a LoRA adapter into its base model and quantize to GGUF, so the
# fine-tune can be benchmarked against the decision gate (see README.md).
#
# Usage:
#   bash merge_and_quantize.sh <base-model> <adapter-dir> [QUANT]
#   bash merge_and_quantize.sh Llama-3.2-3B-Instruct training/adapters/run1 Q4_K_M
#
# Produces training/merged/<name>.<QUANT>.gguf (gitignored). After this,
# point MODEL_PATH at the GGUF, run scripts/profile_memory.py (RAM gate)
# and scripts/eval_swahili.py (quality gate), and only ship if BOTH pass.
#
# Requires: transformers + peft (merge), and a built llama.cpp (convert +
# quantize). This runs offline on a capable machine, never on the kiosk.
set -euo pipefail

BASE="${1:?base model (HF id or path) required}"
ADAPTER="${2:?adapter dir required}"
QUANT="${3:-Q4_K_M}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED_DIR="${REPO_ROOT}/training/merged"
LLAMA_CPP="${LLAMA_CPP:-${REPO_ROOT}/llama.cpp}"
mkdir -p "${MERGED_DIR}"

echo "[1/3] Merging adapter into base (peft)..."
python - "$BASE" "$ADAPTER" "${MERGED_DIR}/merged_hf" <<'PY'
import sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = AutoModelForCausalLM.from_pretrained(base, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print(f"merged -> {out}")
PY

echo "[2/3] Converting merged model to GGUF (f16)..."
python "${LLAMA_CPP}/convert_hf_to_gguf.py" \
    "${MERGED_DIR}/merged_hf" \
    --outfile "${MERGED_DIR}/model.f16.gguf" \
    --outtype f16

echo "[3/3] Quantizing to ${QUANT}..."
"${LLAMA_CPP}/build/bin/llama-quantize" \
    "${MERGED_DIR}/model.f16.gguf" \
    "${MERGED_DIR}/model.${QUANT}.gguf" \
    "${QUANT}"

echo
echo "Done: ${MERGED_DIR}/model.${QUANT}.gguf"
echo "Now run the decision gate:"
echo "  MODEL_PATH=training/merged/model.${QUANT}.gguf python -m scripts.profile_memory   # RAM < 7 GB?"
echo "  python -m scripts.eval_swahili --compare                                          # beats prompt-only?"
echo "Ship the fine-tune ONLY if both pass and English does not regress."
