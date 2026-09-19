#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$HOME/llama.cpp}"
readonly LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_CPP_ROOT/build/bin/llama-server}"
readonly K2_MODEL="${K2_MODEL:-}"
readonly K2_REASONING_BUDGET="${K2_REASONING_BUDGET:-512}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  printf 'ERROR: llama-server is not executable: %s\n' "$LLAMA_SERVER" >&2
  exit 1
fi
if [[ -z "$K2_MODEL" || ! -f "$K2_MODEL" ]]; then
  printf 'ERROR: set K2_MODEL to the first K2-Think-V2 Q6_K GGUF shard\n' >&2
  exit 1
fi
if [[ ! "$K2_REASONING_BUDGET" =~ ^[0-9]+$ ]]; then
  printf 'ERROR: K2_REASONING_BUDGET must be a non-negative integer\n' >&2
  exit 1
fi

exec "$LLAMA_SERVER" \
  --model "$K2_MODEL" \
  --host 127.0.0.1 \
  --port 30000 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --parallel 1 \
  --no-warmup \
  --temp 1 \
  --top-p 1 \
  --top-k 0 \
  --min-p 0 \
  --reasoning-format deepseek \
  --reasoning-budget "$K2_REASONING_BUDGET" \
  --reasoning-budget-message $'\nThe reasoning budget is exhausted. Give only the requested final answer.\n'
