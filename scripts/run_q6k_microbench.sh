#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly BINARY="${Q6K_BINARY:-$PROJECT_DIR/build-microbenchmark/q6k-microbench}"
readonly TIMEOUT_SECONDS=180

if (( EUID == 0 )); then
  printf 'ERROR: refusing to run as root or through sudo\n' >&2
  exit 1
fi
if [[ ! -x "$BINARY" || -L "$BINARY" ]]; then
  printf 'ERROR: isolated benchmark binary is missing; run scripts/build_q6k_microbench.sh\n' >&2
  exit 1
fi

args=()
while (( $# > 0 )); do
  case "$1" in
    --dry-run|--execute)
      args+=("$1")
      shift
      ;;
    --columns|--warmup|--iterations)
      (( $# >= 2 )) || { printf 'ERROR: %s requires an integer value\n' "$1" >&2; exit 1; }
      [[ "$2" =~ ^[0-9]+$ ]] || { printf 'ERROR: %s requires a non-negative integer\n' "$1" >&2; exit 1; }
      args+=("$1" "$2")
      shift 2
      ;;
    *)
      printf 'ERROR: unsupported argument (model paths and server arguments are forbidden): %s\n' "$1" >&2
      exit 1
      ;;
  esac
done

BENCH_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$BENCH_PID" ]] && kill -0 "$BENCH_PID" 2>/dev/null; then
    kill -TERM -- "-$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if (( ${#args[@]} == 0 )); then
  args=(--dry-run)
fi

setsid timeout --signal=TERM --kill-after=5s "${TIMEOUT_SECONDS}s" "$BINARY" "${args[@]}" &
BENCH_PID=$!
wait "$BENCH_PID"
BENCH_PID=""
