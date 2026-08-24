#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly BUILD_DIR="${Q6K_BUILD_DIR:-$PROJECT_DIR/build-microbenchmark}"
readonly LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$HOME/llama.cpp}"

if (( EUID == 0 )); then
  printf 'ERROR: refusing to build as root or through sudo\n' >&2
  exit 1
fi

cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G "Unix Makefiles" \
  -DLLAMA_CPP_ROOT="$LLAMA_CPP_ROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build "$BUILD_DIR" --target q6k-microbench --parallel 4

printf 'Built isolated binary: %s/q6k-microbench\n' "$BUILD_DIR"
