#!/usr/bin/env bash
# Launch llama-server for rl-finetune.
#
# Usage:
#   bash scripts/serve.sh [MODEL_PATH] [PORT]
#
# Examples:
#   bash scripts/serve.sh                                       # auto-detect
#   bash scripts/serve.sh models/Qwen3-1.7B-Q4_K_M.gguf
#   bash scripts/serve.sh models/finetuned.gguf 8080
#
# Or use the CLI:
#   rl serve --model models/finetuned.gguf

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Resolve model path ---
MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  # Find the most recent .gguf in models/
  if [[ -d models ]]; then
    MODEL=$(ls -t models/*.gguf 2>/dev/null | head -1 || true)
  fi
fi

if [[ -z "$MODEL" ]]; then
  echo "Error: no model specified and no .gguf found in models/"
  echo "Usage: bash scripts/serve.sh <model.gguf> [port]"
  exit 1
fi

PORT="${2:-8080}"
CTX_SIZE="${LLAMA_CTX:-4096}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"

echo "Starting llama-server:"
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "  Context:  $CTX_SIZE"
echo "  GPU layers: $GPU_LAYERS"
echo ""

exec llama-server \
  -m "$MODEL" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --n-gpu-layers "$GPU_LAYERS" \
  --cont-batching \
  --log-prefix
