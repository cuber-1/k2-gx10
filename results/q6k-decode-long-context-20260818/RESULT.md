# Q6_K long-context decode validation

Status: **ACCEPTED**. The preregistered primary and independent confirmation
campaigns both passed at every tested KV depth. An independent review reproduced
the statistics and accepted the runtime, safety, order-balance, and provenance
checks.

The comparison isolates the accepted Q6_K prefetch/persistence change: baseline
is the accepted 8-warp-only build (`build-decode-server`), and candidate is the
accepted 8-warp plus 0/128 prefetch build (`build-decode-prefetch-server`). The
candidate GPU code is byte-equivalent to the final DGX Spark package for this
path. This is not an upstream-to-final net comparison.

## Confirmation result

Each depth has ten independent A/B process pairs, exactly balanced 5/5 by which
build ran first. Every process used one model load, fixed `n_ctx=8192`, FA on,
f16 K/V, CUDA graphs, one excluded full-depth 128-token warmup, and one measured
128-token synchronized decode segment. Positive change is exact-nanosecond
throughput speedup: `100 * (baseline_ns / candidate_ns - 1)`.

| Timed KV band | Paired median throughput change | 95% bootstrap CI | Wins | Decision |
|---|---:|---:|---:|---|
| 0..127 | +11.7542% | [+11.5729%, +11.9707%] | 10/10 | persistent |
| 2048..2175 | +11.5874% | [+11.4104%, +11.7400%] | 10/10 | persistent |
| 4096..4223 | +11.5457% | [+11.3421%, +11.7453%] | 10/10 | persistent |
| 7168..7295 | +11.2610% | [+11.1505%, +11.6203%] | 10/10 | persistent |

Both build-order strata were positive at every depth. All 20 confirmation
processes exited zero; all 20 telemetry records were free of contention and
safety violations. The longest-depth absolute time saving retained 99.86% of
the shallow-depth saving, so the preregistered Nsight Systems dilution fallback
was not triggered.

The primary campaign also passed all depths, with paired medians of +11.9985%,
+11.6166%, +11.4940%, and +10.9621% from shallowest to deepest. The reversed
depth/order confirmation above is the cleaner final estimate.

## Record

- Method and gates: `PREREGISTRATION.md`
- Primary report: `RESULT-initial.md`
- Confirmation report: `RESULT-repeat.md`
- Machine-readable summaries: `summary-initial.json`, `summary-repeat.json`
- Pair-level data: `paired-invocations-initial.csv`,
  `paired-invocations-repeat.csv`
- Invocation-level data: `invocations-initial.csv`, `invocations-repeat.csv`
- Raw evidence: `raw/`, `raw-repeat/`
- Safety telemetry: `telemetry/`, `telemetry-repeat/`
- Frozen hashes and environment: `EXPECTED_SHA256.txt`, `provenance/`

Key output hashes:

- `RESULT-initial.md`: `d7210e75042633dc31f4e42878c04881535a6bd95adea8c7abb837b19fadda51`
- `summary-initial.json`: `5d489f5ff91ec78dcca9c80e358fc1d431ac2bbdf1a713a5a6c01fb92f618847`
- `RESULT-repeat.md`: `a079c1f3330066822c04390d1d69cae603462df944595abb9cb6e136d8bfc1a1`
- `summary-repeat.json`: `041e354a2ab738f0a537462fee77f4a571632fa40a87caa365cf6d3b91de7115`

