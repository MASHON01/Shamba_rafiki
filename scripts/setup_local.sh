#!/usr/bin/env bash
#
# Shamba Rafiki - local bootstrap.
#
# Gets a fresh checkout to a runnable state on this machine:
#   1. create a Python virtualenv and install dependencies
#   2. create the runtime directories (data/, models/, logs/)
#   3. clone + build llama.cpp (the inference engine) into ./llama.cpp
#   4. run a smoke test (all 98 tests, on fakes - no model needed)
#
# llama.cpp is NOT committed to this repo (see .gitignore); it is built
# here, on THIS machine, so the binary matches the local CPU. That is
# what makes a clean checkout reproducible on the judge's target
# machine.
#
# Usage:
#   ./scripts/setup_local.sh              # full setup
#   ./scripts/setup_local.sh --llama-only # only clone + build llama.cpp
#   ./scripts/setup_local.sh --no-llama   # skip llama.cpp (deps + tests only)
#
# Run from the repository root.

set -euo pipefail

# Resolve the repo root from this script's location, so it works from
# anywhere (matching the CWD-independent app paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

LLAMA_ONLY=false
NO_LLAMA=false
for arg in "$@"; do
  case "$arg" in
    --llama-only) LLAMA_ONLY=true ;;
    --no-llama)   NO_LLAMA=true ;;
    *) echo "Unknown option: $arg" ; exit 1 ;;
  esac
done

log() { printf "\n\033[1;32m==>\033[0m %s\n" "$1"; }

# --- 1. Python venv + dependencies ------------------------------------
setup_python() {
  log "Creating virtualenv (.venv) and installing dependencies"
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e ".[dev]"
  log "Python environment ready"
}

# --- 2. Runtime directories -------------------------------------------
setup_dirs() {
  log "Creating runtime directories"
  mkdir -p data/raw_documents data/processed_documents \
           data/vector_store data/embeddings data/reports \
           models logs
  touch logs/.gitkeep
}

# --- 3. llama.cpp (clone + build) -------------------------------------
setup_llama() {
  log "Setting up llama.cpp"
  if [ ! -d "llama.cpp" ]; then
    log "Cloning llama.cpp"
    git clone https://github.com/ggerganov/llama.cpp.git
  else
    log "llama.cpp already present - pulling latest"
    git -C llama.cpp pull --ff-only || true
  fi

  log "Building llama.cpp (this compiles on THIS machine - may take a while)"
  cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build llama.cpp/build --config Release -j"$(nproc 2>/dev/null || echo 4)"

  if [ -f "llama.cpp/build/bin/llama-server" ]; then
    log "llama-server built: llama.cpp/build/bin/llama-server"
  else
    echo "WARNING: llama-server binary not found where expected."
    echo "Check the llama.cpp build output above."
  fi

  echo
  echo "Next: download a GGUF model into ./models/ (e.g. llama.gguf), then start"
  echo "the server, e.g.:"
  echo "  ./llama.cpp/build/bin/llama-server -m models/llama.gguf --host 0.0.0.0 --port 8080 -c 4096"
}

# --- 4. Smoke test -----------------------------------------------------
smoke_test() {
  log "Running the test suite (all on fakes - no model required)"
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || true
  if python -m pytest -q; then
    log "Smoke test passed - the backend is runnable."
  else
    echo "Smoke test FAILED - see output above."
    exit 1
  fi
}

# --- orchestrate -------------------------------------------------------
if [ "$LLAMA_ONLY" = true ]; then
  setup_llama
  exit 0
fi

setup_python
setup_dirs
if [ "$NO_LLAMA" = false ]; then
  setup_llama
else
  log "Skipping llama.cpp (--no-llama)"
fi
smoke_test

log "Setup complete."
echo
echo "To run the backend (needs llama-server running separately):"
echo "  make run          # dev, autoreload"
echo "  make run-prod     # production settings"
echo
echo "See docs/DEPLOYMENT.md for the full kiosk setup."