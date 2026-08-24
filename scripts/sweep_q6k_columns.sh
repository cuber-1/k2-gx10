#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly BINARY="${Q6K_BINARY:-$PROJECT_DIR/build-microbenchmark/q6k-microbench}"
readonly RUNNER="$PROJECT_DIR/scripts/run_q6k_microbench.sh"
readonly RESULTS_ROOT="$PROJECT_DIR/results/q6k-column-sweep"
readonly WARMUP="${Q6K_SWEEP_WARMUP:-3}"
readonly ITERATIONS="${Q6K_SWEEP_ITERATIONS:-10}"
readonly TIMEOUT_SECONDS=180
readonly SWEEP_COLUMNS=(25 32 64 100 128 256 512 768 1024 1280 1536 1792 2048)

WITH_NSYS=false
case "${1:-}" in
  "") ;;
  --with-nsys) WITH_NSYS=true ;;
  *) printf 'Usage: %s [--with-nsys]\n' "$0" >&2; exit 2 ;;
esac
(( $# <= 1 )) || { printf 'Usage: %s [--with-nsys]\n' "$0" >&2; exit 2; }

if (( EUID == 0 )); then
  printf 'ERROR: refusing to run as root or through sudo\n' >&2
  exit 1
fi
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { printf 'ERROR: Q6K_SWEEP_WARMUP must be an integer\n' >&2; exit 1; }
[[ "$ITERATIONS" =~ ^[0-9]+$ ]] || { printf 'ERROR: Q6K_SWEEP_ITERATIONS must be an integer\n' >&2; exit 1; }
(( WARMUP <= 20 )) || { printf 'ERROR: warmup exceeds benchmark limit 20\n' >&2; exit 1; }
(( ITERATIONS >= 1 && ITERATIONS <= 100 )) || {
  printf 'ERROR: iterations must be in 1..100\n' >&2
  exit 1
}
[[ -x "$BINARY" && ! -L "$BINARY" ]] || {
  printf 'ERROR: isolated benchmark is missing; run scripts/build_q6k_microbench.sh\n' >&2
  exit 1
}
[[ -x "$RUNNER" && ! -L "$RUNNER" ]] || {
  printf 'ERROR: guarded benchmark runner is missing\n' >&2
  exit 1
}
if [[ "$WITH_NSYS" == true ]]; then
  command -v nsys >/dev/null || { printf 'ERROR: nsys is required for --with-nsys\n' >&2; exit 1; }
fi

run_id="$(date +%Y%m%d-%H%M%S)"
result_dir="$RESULTS_ROOT/$run_id"
mkdir -p "$result_dir"
csv="$result_dir/sweep.csv"
printf '%s\n' 'columns,predicted_j,guarded_bytes,cpu_ms,cuda_median_ms,cuda_min_ms,cuda_max_ms,nmse,actual_kernel_family,actual_j' >"$csv"

for columns in "${SWEEP_COLUMNS[@]}"; do
  printf '\n=== columns=%s ===\n' "$columns"
  log="$result_dir/n-${columns}.log"
  output="$("$RUNNER" --execute --columns "$columns" --warmup "$WARMUP" --iterations "$ITERATIONS")"
  printf '%s\n' "$output" | tee "$log"

  predicted_j="$(sed -n 's/^MMQ prediction:.*J=\([0-9][0-9]*\).*/\1/p' <<<"$output")"
  guarded="$(sed -n 's/^  GUARDED TOTAL[[:space:]]*\([0-9][0-9]*\) B.*/\1/p' <<<"$output")"
  cpu_ms="$(sed -n 's/^CPU reference time: \([0-9.][0-9.]*\) ms$/\1/p' <<<"$output")"
  cuda_median="$(sed -n 's/^CUDA timing: median=\([0-9.][0-9.]*\) ms.*/\1/p' <<<"$output")"
  cuda_min="$(sed -n 's/^CUDA timing:.* min=\([0-9.][0-9.]*\) ms.*/\1/p' <<<"$output")"
  cuda_max="$(sed -n 's/^CUDA timing:.* max=\([0-9.][0-9.]*\) ms.*/\1/p' <<<"$output")"
  nmse="$(sed -n 's/^correctness: NMSE=\([^[:space:]]*\).*/\1/p' <<<"$output")"
  [[ -n "$predicted_j" && -n "$guarded" && -n "$cpu_ms" && -n "$cuda_median" &&
     -n "$cuda_min" && -n "$cuda_max" && -n "$nmse" ]] || {
    printf 'ERROR: could not parse benchmark output for columns=%s\n' "$columns" >&2
    exit 1
  }

  actual_family=""
  actual_j=""
  if [[ "$WITH_NSYS" == true ]]; then
    trace_base="$result_dir/n-${columns}"
    trace_log="$result_dir/n-${columns}-nsys.log"
    stats="$result_dir/n-${columns}-nsys-stats.txt"
    timeout --signal=TERM --kill-after=5s "${TIMEOUT_SECONDS}s"       nsys profile --trace=cuda --sample=none --cpuctxsw=none       --force-overwrite=true --output "$trace_base"       "$BINARY" --execute --columns "$columns" >"$trace_log" 2>&1
    nsys stats --report cuda_gpu_kern_sum "$trace_base.nsys-rep" >"$stats"
    actual_family="$(grep -o -m1 'mul_mat[_a-zA-Z0-9]*' "$stats" || true)"
    actual_j="$(sed -n 's/.*mul_mat_q<.*(int)\([0-9][0-9]*\).*/\1/p' "$stats" | head -1)"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'     "$columns" "$predicted_j" "$guarded" "$cpu_ms" "$cuda_median"     "$cuda_min" "$cuda_max" "$nmse" "$actual_family" "$actual_j" >>"$csv"
done

printf '\nSweep complete\nResults: %s\nCSV: %s\n' "$result_dir" "$csv"
