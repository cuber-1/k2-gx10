#!/usr/bin/env bash
set -euo pipefail

readonly HOST="127.0.0.1"
readonly PORT="30000"
readonly REASONING_BUDGET="256"
readonly GRAPH_NODE_PROFILE_TIMEOUT_SECONDS="900"
readonly GRAPH_NODE_REQUEST_TIMEOUT_SECONDS="600"

GRAPH_NODES=false

usage() {
  cat <<EOF
Usage: $0 [--graph-nodes]

With no arguments, run the existing graph-level warmup and measured capture.
With --graph-nodes, collect CUDA graph node activities for one bounded four-token request.
EOF
}

case "${1:-}" in
  "") ;;
  --graph-nodes)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    GRAPH_NODES=true
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROFILE_DIR="$PROJECT_DIR/profiles/nsys"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
if [[ "$GRAPH_NODES" == true ]]; then
  RUN_ID="${RUN_ID}-graph-nodes"
fi
REPORT_BASE="$PROFILE_DIR/k2-${RUN_ID}"
REPORT_FILE="${REPORT_BASE}.nsys-rep"
STATS_FILE="${REPORT_BASE}-stats.txt"
WARMUP_LOG="${REPORT_BASE}-warmup.txt"
MEASURED_LOG="${REPORT_BASE}-measured.txt"

PROFILE_PID=""

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM

  if [[ -n "$PROFILE_PID" ]] && kill -0 "$PROFILE_PID" 2>/dev/null; then
    # The profiler and its server run in a new session/process group. Signal only
    # that group, never an unrelated process that may be using the same port.
    kill -TERM -- "-$PROFILE_PID" 2>/dev/null || true
    wait "$PROFILE_PID" 2>/dev/null || true
  fi

  exit "$exit_status"
}
trap cleanup EXIT INT TERM

required_commands=(nsys ss curl setsid)
if [[ "$GRAPH_NODES" == true ]]; then
  required_commands+=(timeout)
fi

for command in "${required_commands[@]}"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $command" >&2
    exit 1
  fi
done

if ss -H -ltn "sport = :$PORT" | grep -q .; then
  echo "ERROR: port $PORT is already occupied; no profiling session was started." >&2
  ss -H -ltnp "sport = :$PORT" >&2 || true
  exit 1
fi

mkdir -p "$PROFILE_DIR"

echo "Starting K2 server under Nsight Systems on $HOST:$PORT"
echo "Report: $REPORT_FILE"

cd "$PROJECT_DIR"
if [[ "$GRAPH_NODES" == true ]]; then
  setsid timeout --signal=TERM --kill-after=10s "${GRAPH_NODE_PROFILE_TIMEOUT_SECONDS}s" \
    nsys profile \
      --force-overwrite=true \
      --output "$REPORT_BASE" \
      --trace=cuda \
      --cuda-graph-trace=node \
      --sample=none \
      --cpuctxsw=none \
      env K2_REASONING_BUDGET="$REASONING_BUDGET" \
      ./run-k2-server.sh &
else
  setsid nsys profile \
    --force-overwrite=true \
    --output "$REPORT_BASE" \
    --trace=cuda,nvtx,osrt \
    env K2_REASONING_BUDGET="$REASONING_BUDGET" \
    ./run-k2-server.sh &
fi
PROFILE_PID=$!

ready=false
for _ in $(seq 1 120); do
  if ! kill -0 "$PROFILE_PID" 2>/dev/null; then
    echo "ERROR: profiler or server exited before becoming ready." >&2
    wait "$PROFILE_PID" || true
    exit 1
  fi

  if curl --silent --fail --max-time 1 "http://$HOST:$PORT/health" >/dev/null; then
    ready=true
    break
  fi
  sleep 1
done

if [[ "$ready" != true ]]; then
  echo "ERROR: server did not become ready within 120 seconds." >&2
  exit 1
fi

if [[ "$GRAPH_NODES" == true ]]; then
  echo "Running one bounded four-token graph-node request"
  curl --silent --show-error --fail \
    --max-time "$GRAPH_NODE_REQUEST_TIMEOUT_SECONDS" \
    --header "Content-Type: application/json" \
    --data-binary '{"model":"K2 Think V2","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":4,"temperature":0,"seed":42,"reasoning_format":"deepseek","reasoning_budget_tokens":1}' \
    "http://$HOST:$PORT/v1/chat/completions" >"$MEASURED_LOG"
else
  echo "Running warm-up request"
  K2_BASE_URL="http://$HOST:$PORT" \
  K2_TEST_REASONING_BUDGET="$REASONING_BUDGET" \
  K2_TEST_MAX_TOKENS="$((REASONING_BUDGET + 64))" \
    ./client_test.py >"$WARMUP_LOG" 2>&1

  echo "Running measured request"
  K2_BASE_URL="http://$HOST:$PORT" \
  K2_TEST_REASONING_BUDGET="$REASONING_BUDGET" \
  K2_TEST_MAX_TOKENS="$((REASONING_BUDGET + 64))" \
    ./client_test.py >"$MEASURED_LOG" 2>&1
fi

echo "Stopping this profiling session"
kill -INT -- "-$PROFILE_PID" 2>/dev/null || true
wait "$PROFILE_PID" || true
PROFILE_PID=""

if [[ ! -f "$REPORT_FILE" ]]; then
  echo "ERROR: Nsight Systems did not create $REPORT_FILE" >&2
  exit 1
fi

echo "Writing stats summary: $STATS_FILE"
nsys stats "$REPORT_FILE" >"$STATS_FILE"

echo "Profiling complete"
echo "Report: $REPORT_FILE"
echo "Stats:  $STATS_FILE"
