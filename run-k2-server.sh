#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/home/dvijraicha/llama.cpp/build/bin/llama-server}"
K2_MODEL="${K2_MODEL:-/home/dvijraicha/.cache/huggingface/hub/models--benjaminradio--K2-Think-V2-GGUF/snapshots/3064ec56b7c735f4f133aa10cfcca3ef3bd718f7/K2-Think-V2-Q6_K-00001-of-00004.gguf}"
K2_REASONING_BUDGET="${K2_REASONING_BUDGET:-512}"

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
