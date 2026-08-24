#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RESULT_DIR="${COMBINED_DECODE_RESULT_DIR:-$ROOT/results/q6k-decode-combined-20260824}"
readonly BASELINE="${COMBINED_DECODE_BASELINE:-/home/dvijraicha/llama.cpp/build/bin/llama-bench}"
readonly FINAL="${COMBINED_DECODE_FINAL:-$ROOT/build-decode-prefetch-server/bin/llama-bench}"
readonly MODEL="${K2_MODEL:-/home/dvijraicha/.cache/huggingface/hub/models--benjaminradio--K2-Think-V2-GGUF/snapshots/3064ec56b7c735f4f133aa10cfcca3ef3bd718f7/K2-Think-V2-Q6_K-00001-of-00004.gguf}"
readonly HARNESS="$ROOT/results/q6k-decode-long-context-20260818/libk2_fixed_ctx_work.so"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ ! -e "$RESULT_DIR" ]] || fail "result directory already exists: $RESULT_DIR"
[[ -x "$BASELINE" ]] || fail "baseline binary is not executable: $BASELINE"
[[ -x "$FINAL" ]] || fail "final binary is not executable: $FINAL"
[[ -f "$MODEL" ]] || fail "model is missing: $MODEL"
[[ -f "$HARNESS" ]] || fail "fixed-context harness is missing: $HARNESS"
[[ -z "${LD_PRELOAD+x}" ]] || fail "inherited LD_PRELOAD is set"
[[ -z "${LD_LIBRARY_PATH+x}" ]] || fail "inherited LD_LIBRARY_PATH is set"

exec 9>/tmp/k2-gx10-gpu.lock
flock -n 9 || fail "global GPU lock is held"

[[ -z "$(ollama ps 2>/dev/null | tail -n +2 | sed '/^[[:space:]]*$/d')" ]] || fail "Ollama has a loaded model"
if pgrep -x llama-bench >/dev/null || pgrep -x llama-server >/dev/null || pgrep -x ncu >/dev/null || pgrep -x nsys >/dev/null; then
    fail "another benchmark, server, or profiler is active"
fi

mkdir -p "$RESULT_DIR/raw" "$RESULT_DIR/provenance"
date --iso-8601=seconds > "$RESULT_DIR/STARTED"
sha256sum "$BASELINE" "$FINAL" "$HARNESS" > "$RESULT_DIR/provenance/artifact-sha256.txt"
sha256sum \
    /home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/mmvq.cu \
    "$ROOT/vendor/llama-decode-prefetch-server/ggml/src/ggml-cuda/mmvq.cu" \
    > "$RESULT_DIR/provenance/source-sha256.txt"
find -L "$(dirname "$MODEL")" -maxdepth 1 -type f -name 'K2-Think-V2-Q6_K-*.gguf' -printf '%p %s bytes\n' | sort \
    > "$RESULT_DIR/provenance/model-files.txt"
nvidia-smi -q > "$RESULT_DIR/provenance/nvidia-smi-before.txt"
ldd "$BASELINE" > "$RESULT_DIR/provenance/ldd-baseline.txt"
ldd "$FINAL" > "$RESULT_DIR/provenance/ldd-final.txt"

run_one() {
    local pair="$1" variant="$2" binary="$3" stem
    stem="$RESULT_DIR/raw/pair-$(printf '%02d' "$pair")-$variant"
    echo "$(date --iso-8601=seconds) pair=$pair variant=$variant binary=$binary" | tee -a "$RESULT_DIR/provenance/order.log"
    timeout --signal=TERM --kill-after=15s 1200s \
        env LD_PRELOAD="$HARNESS" \
        "$binary" -m "$MODEL" -ngl 99 -p 0 -n 128 -d 0 \
        -b 2048 -ub 512 -ctk f16 -ctv f16 -fa on -t 20 -r 2 --delay 1 -o json -v \
        > "$stem.json" 2> "$stem.stderr"
    python3 -m json.tool "$stem.json" >/dev/null
    grep -q 'K2_FIXED_CONTEXT requested=8192 effective=8192' "$stem.stderr" \
        || fail "fixed context was not verified for pair=$pair variant=$variant"
    grep -q 'K2_FIXED_WORK ' "$stem.stderr" \
        || fail "fixed work was not reported for pair=$pair variant=$variant"
}

for pair in 1 2 3 4 5; do
    if (( pair % 2 == 1 )); then
        run_one "$pair" baseline "$BASELINE"
        run_one "$pair" final "$FINAL"
    else
        run_one "$pair" final "$FINAL"
        run_one "$pair" baseline "$BASELINE"
    fi
    baseline_work="$(grep '^K2_FIXED_WORK ' "$RESULT_DIR/raw/pair-$(printf '%02d' "$pair")-baseline.stderr")"
    final_work="$(grep '^K2_FIXED_WORK ' "$RESULT_DIR/raw/pair-$(printf '%02d' "$pair")-final.stderr")"
    [[ "$baseline_work" == "$final_work" ]] || fail "fixed-work mismatch in pair $pair"
done

python3 "$ROOT/scripts/analyze_combined_decode_ab.py" "$RESULT_DIR" | tee "$RESULT_DIR/analysis.stdout"
nvidia-smi -q > "$RESULT_DIR/provenance/nvidia-smi-after.txt"
date --iso-8601=seconds > "$RESULT_DIR/COMPLETED"
