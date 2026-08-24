# Exact-size Q6_K field-SoA decode result

Status: **REJECTED**. The layout is exact and safe, but it did not pass the
predeclared performance gate. The accepted production patch remains unchanged.

## What changed

The isolated candidate repacks complete K2 FFN row groups without padding:

- K=8192: 2 rows/group, 64 Q6 blocks, 13,440 bytes = 105 cache lines;
- K=28672: 4 rows/group, 448 Q6 blocks, 94,080 bytes = 735 cache lines.

Each group stores all 128-byte `ql` fields first, followed by every unchanged
82-byte `qh + scales + d` tail. Total matrix size and every quantized byte are
unchanged. The candidate adds exact-shape eight-warp kernels and is guarded by
`K2_Q6_SOA=1`, Q6_K, N=1, cc1210, and the two K2 FFN shapes. Unsupported calls
are refused while the experimental environment variable is active, preventing
ordinary row-major weights from being silently misread.

Source anchors are `mmvq.cu:747-867` for mapping/dot/kernel code and
`mmvq.cu:1433-1463` for prototype dispatch. The result-local host repacker and
round-trip verifier are at `src/q6k-ffn-microbench.cpp:110-178`.

## Static and correctness gates

- Fused K=8192: 48 registers, 2,816 B shared, zero stack/local.
- Nonfused K=8192 and K=28672: 40 registers, 1,920 B shared, zero stack/local.
- Existing accepted fused/nonfused kernel instruction encodings and resources
  remain exact matches.
- The repacker verified all three deterministic matrices byte for byte without
  allocating a third full copy.
- Accepted and repacked GPU outputs are byte-identical: SHA-256
  `d5868362d8792632ee78479c217b8bbaa4dc5697691250db9ac6faf8b985e26b`.
- CPU/CUDA NMSE is `0.000232741002` (gate `0.0005`).
- Compute Sanitizer reports zero errors.
- The guarded repacked harness total is 2,005,008,384 B (1,912.125 MiB), below
  the 2 GiB hard limit.

## Measured performance

Ten balanced process-level pairs used 20 warmups and 100 timed full-FFN graph
iterations per invocation. Positive values mean the repacked candidate is
faster.

| Metric | Result |
|---|---:|
| Baseline median | 2.649022 ms |
| Candidate median | 2.616062 ms |
| Paired median throughput change | +0.769156% |
| Paired IQR | [-0.878004%, +2.320647%] |
| Pair wins | 6/10 |
| Seeded bootstrap 95% interval | [-0.963737%, +2.557247%] |

The required gate was at least +2%, at least 8/10 wins, and an interval wholly
above zero. All three conditions failed. No repeat, profiler run, GGUF loader,
full-model conversion, or production integration was performed.

## Decision

This field separation makes the 128-byte `ql` loads regular, but the unchanged
82-byte tail remains irregular and the extra group/field address arithmetic is
not repaid reliably at full-FFN scale. Keep the accepted row-major 8-warp plus
0/128 L2-prefetch patch. A further repack attempt would need a genuinely tiled
lane-to-data mapping (for example a GPU-native AoSoA consumed cooperatively),
not another address-only rearrangement.

A subsequent stronger prototype separated `qh`, scales, and deltas as well; it
also failed at +0.614% with 6/10 wins. See
`results/q6k-decode-q6-full-soa/RESULT.md`.

Raw evidence is in `correctness/`, `static/candidate/`, and `timing-set1/`.
