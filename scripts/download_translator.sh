#!/usr/bin/env bash
#
# Build the offline Swahili translator: NLLB-200-distilled-600M converted
# to CTranslate2 (int8) so translation runs on CPU with no PyTorch resident
# at serve time.
#
# One-time step. Needs internet (downloads the model from Hugging Face) and:
#     pip install ctranslate2 sentencepiece
# transformers is already installed via sentence-transformers.
#
#     bash scripts/download_translator.sh
#
# The app works without this; it just falls back to the Swahili-prompt path
# until the model exists. Once built, restart the backend and Swahili
# questions are answered English-first then translated back.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_MODEL="facebook/nllb-200-distilled-600M"
OUT_DIR="$REPO_ROOT/models/nllb-200-distilled-600M-ct2"

if [ -f "$OUT_DIR/model.bin" ]; then
  echo "Translator already built at $OUT_DIR, skipping."
  exit 0
fi

if ! command -v ct2-transformers-converter >/dev/null 2>&1; then
  echo "ERROR: ct2-transformers-converter not found. Run: pip install ctranslate2" >&2
  exit 1
fi

echo "Converting $HF_MODEL to CTranslate2 (int8) at:"
echo "  $OUT_DIR"
ct2-transformers-converter \
  --model "$HF_MODEL" \
  --output_dir "$OUT_DIR" \
  --quantization int8 \
  --force

echo "Saving the tokenizer alongside the model ..."
python - "$HF_MODEL" "$OUT_DIR" <<'PY'
import sys
from transformers import AutoTokenizer

model, out = sys.argv[1], sys.argv[2]
AutoTokenizer.from_pretrained(model).save_pretrained(out)
print("tokenizer saved to", out)
PY

echo "Done. Restart the backend; Swahili now uses fluent NLLB translation."
