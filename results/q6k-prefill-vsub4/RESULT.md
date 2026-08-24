# Result: DGX Spark Q6_K MMQ `vsub4`

Decision: **rejected**. The cc1210-only non-saturating packed subtract is
bit-correct and materially simplifies SASS, but normal unprofiled primary
timing shows a reproducible 2.28% regression rather than an improvement.

## Static evidence

At `mul_mat_q<Q6_K,128,false>`, the candidate removes all 128 saturation-tail
`PRMT 0xba98` operations and reduces the instruction count from 8,744 to 8,432.
It retains 32 `PRMT 0x8880`, 512 IMMA, 1,280 FFMA, and identical global,
shared, barrier, and local-memory counts. Resources remain 255 registers,
64-byte stack, 32 LDL, and 27 STL. J32 remains stack-free and all Stream-K
fixup encodings are identical. This confirms that the tested change was real
and isolated, not a canonical compiler no-op.

## Correctness

All seven required M/K/N shapes pass CPU-reference NMSE with values between
2.427e-5 and 2.448e-5 versus the 5e-4 threshold. Every baseline/candidate raw
GPU output is byte-identical and has a matching SHA256. Full metrics and hashes
are in `correctness/SUMMARY.md`.

## Primary timing

Shape: M=8192, K=8192, N=512. Ten alternating baseline/candidate pairs, each
with 20 warmups and 100 timed iterations:

| metric | baseline | candidate |
|---|---:|---:|
| median | 1.9765 ms | 2.0215 ms |
| Q25 | 1.9670 ms | 2.01875 ms |
| Q75 | 1.9810 ms | 2.0250 ms |
| IQR | 0.0140 ms | 0.00625 ms |

Paired median speedup was -2.276759% (a candidate slowdown), with paired
Q25/Q75 of -2.783427%/-1.829884%, 0/10 candidate wins, and deterministic
100,000-resample bootstrap 95% CI [-3.002807%, -1.710643%]. Seed was
`0x4b325136` and percentiles use type-7 linear interpolation.

The result misses every promotion gate: required median speedup >=2%, >=8/10
wins, and CI lower bound >0. The regression despite fewer instructions implies
that instruction-count reduction alone does not improve this kernel's schedule
on GB10; the evidence does not isolate a single lower-level cause.

## Disposition

- No control-shape timing or fresh repeat was run because the primary gate
  failed decisively.
- No Nsight Compute, full-model benchmark, or model load was run.
- Preserve as a rejected prefill idea. Do not combine this substitution into
  the accepted tree absent new scheduling evidence that changes the rationale.
- Accepted source/build artifacts and `/home/dvijraicha/llama.cpp` remain
  untouched. The rejected candidate stays isolated at
  `vendor/llama-prefill-vsub4` and `build-prefill-vsub4`.
