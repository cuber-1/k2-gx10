# Result: forced cuBLAS for Q6_K prefill

Decision: **rejected**. Correctness passed the predeclared synthetic gates,
but forced cuBLAS was 18.71% slower at the production N=1024 microbenchmark
shape and lost all five warmed pairs. This exceeds the 3% early-rejection
threshold, so the experiment stopped without extended timing or profiling.

## Correctness

The initial finite-output and absolute-NMSE gate was recorded before candidate
configuration. Before any GPU execution, the gate was strengthened to require
candidate NMSE and maximum absolute error each remain within 4x the matching
baseline shape; `HYPOTHESIS.md` contains the amended pre-run gate. No numerical
threshold was relaxed after observing results.

All outputs were finite and all absolute CPU-reference NMSE values were below
5e-4. Candidate error stayed below the required 4x same-shape baseline error:

| N | baseline NMSE | candidate NMSE | NMSE ratio | baseline max abs | candidate max abs | max-abs ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2.4474707e-05 | 2.63907097e-05 | 1.0783x | 0.23747611 | 0.435512543 | 1.8339x |
| 512 | 2.44736786e-05 | 2.63909398e-05 | 1.0783x | 0.23747611 | 0.435512543 | 1.8339x |
| 25 | 2.42734665e-05 | 1.7612007e-05 | 0.7256x | 0.203239441 | 0.17726326 | 0.8722x |

These are synthetic microbenchmark checks only. Full-model deterministic
logits or quality validation was intentionally not run after the performance
rejection.

## Warmed N=1024 timing

Shape: M=28672, K=8192, N=1024. Five alternating baseline/candidate pairs,
each with 20 warmups and 30 timed iterations:

| pair | baseline ms | candidate ms | candidate slowdown |
|---:|---:|---:|---:|
| 1 | 8.860 | 10.644 | 20.14% |
| 2 | 9.022 | 10.717 | 18.79% |
| 3 | 9.001 | 10.614 | 17.92% |
| 4 | 9.128 | 10.836 | 18.71% |
| 5 | 9.407 | 10.684 | 13.57% |

Baseline paired median was 9.022 ms; candidate paired median was 10.684 ms.
Median paired slowdown was 18.711656%, with 0/5 candidate wins. This is a
clear early rejection rather than a marginal or profiler-derived conclusion.

## Scope and disposition

- Candidate source remained byte-identical to the accepted source; only the
  built-in CMake dispatch option changed.
- Runtime evidence confirms the candidate actually forced cuBLAS and used the
  default F16 arithmetic setting.
- No 10-pair extension, N=512 timing, fresh repeat, Nsight Systems, Nsight
  Compute, model load, or full-model run was performed after rejection.
- Preserve this as a rejected large-N prefill idea. It should not be repeated
  unless a materially different cuBLAS arithmetic mode or conversion strategy
  is justified by new evidence.

Raw correctness logs and metrics are under `correctness/`; alternating timing
logs, telemetry, hashes, and paired statistics are under `timing/`.
