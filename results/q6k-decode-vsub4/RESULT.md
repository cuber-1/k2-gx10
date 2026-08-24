# DGX Spark Q6_K MMVQ `__vsub4` result

Decision: **REJECTED — correct and cheaper SASS, but no measured fused win.**

## Change and build gate

The isolated candidate changes only the proven Q6_K MMVQ packed subtract in
`vecdotq.cuh`, guarded to exact CUDA architecture 1210 with the saturating
operation retained for HIP, MUSA, and all other CUDA architectures.

Exact-symbol cubin/SASS inspection passed:

- fused resources unchanged: 46 registers, 2816 B static shared, zero
  local/stack; nonfused registers improved 48 to 44;
- fused DP4A count remained four;
- zero `LDL`/`STL` spill instructions;
- fused encoded instructions fell from 392 to 368 as the saturation tail was
  replaced by `IADD3` plus `LOP3.LUT` sign correction.

Full exact-symbol evidence and device facts are in `DEVICE_RESOURCE.md` and
`exact-cubin/`.

## Correctness

All candidate CPU-reference checks passed:

- N=1 nonfused: NMSE 2.46054932e-05;
- N=1 fused SwiGLU: NMSE 3.25271543e-05;
- nonfused N=2 through N=8: NMSE 2.40357088e-05 to 2.43782147e-05.

For N=1 fused/nonfused and N=2, baseline and candidate printed identical NMSE
and max-absolute-error summaries. The current harness does not expose raw GPU
outputs or hashes, so a direct candidate-versus-baseline bitwise comparison was
not available without modifying the shared harness. Its fixed M=28672 is
divisible by the row tile and cannot exercise a small row-tail case. Both facts
were recorded rather than expanding the experiment's source scope.

## Unprofiled timing

Each measurement used 20 warmups and 100 iterations. Ten A/B pairs were
interleaved with alternating first build. Positive paired percentage means the
candidate was faster. Spread is across the ten per-invocation medians; raw
per-invocation min/max and logs are preserved under `timing-*-set1/`.

| mode | baseline median | baseline IQR | candidate median | candidate IQR | paired median | paired IQR | wins | bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused N=1 | 1.9005 ms | 0.01775 ms | 1.9085 ms | 0.02550 ms | -0.3157% | 1.0730 pp | 4/10 | [-1.4763%, +0.5738%] |
| nonfused N=1 | 0.8965 ms | 0.00550 ms | 0.8905 ms | 0.02775 ms | +0.2241% | 3.1748 pp | 6/10 | [-1.1123%, +2.5612%] |

Intervals are deterministic 100,000-resample percentile bootstraps of the ten
paired percentage changes' median. The generator seed was `0x4b325136`; the
reported endpoints use the 2.5th and 97.5th percentiles with linear
interpolation between adjacent sorted resamples. The fused result misses every provisional
acceptance requirement: it is below +1.0%, has only 4/10 wins, and its interval
includes zero. The nonfused interval also includes both zero and a regression
larger than 1%.

No fresh repeat was run because only a winning first set qualifies for repeat.
No full-model or Nsight Compute run was performed. The isolated candidate tree
is intentionally retained and clearly marked rejected.
