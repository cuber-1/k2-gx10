# Exact-size full-field Q6_K SoA decode result

Status: **REJECTED**. The complete four-field layout is exact and safe, but
timing remains neutral and unstable. The production patch is unchanged.

## Layout and implementation

This is the stronger follow-up to the partial field-SoA experiment. Within each
complete K2 row group it stores separate arrays for every Q6_K field:

- `ql`: 128 bytes/block, each record 128-byte aligned;
- `qh`: 64 bytes/block, each record 64-byte aligned;
- scales: 16 bytes/block, each record 16-byte aligned;
- delta: 2 bytes/block.

K=8192 uses 64 blocks/13,440-byte groups; K=28672 uses 448
blocks/94,080-byte groups. Every array base is 128-byte aligned and the group
still occupies exactly 210 bytes per source Q6 block. No padding, value,
precision, arithmetic, or reduction-order change is made.

The isolated source is `vendor/llama-decode-q6-full-soa`. Its prototype path is
limited to cc1210, Q6_K, N=1, exact K2 FFN shapes, and
`K2_Q6_FULL_SOA=1`. Ordinary model files are not supported or modified.

## Gates

- All three host matrices pass byte-exact field reconstruction.
- Accepted/candidate GPU output is byte-identical, SHA-256
  `d5868362d8792632ee78479c217b8bbaa4dc5697691250db9ac6faf8b985e26b`.
- CPU/CUDA NMSE is `0.000232741002` (gate `0.0005`).
- Compute Sanitizer reports zero errors.
- Fused kernel: 48 registers, 2,816 B shared, zero stack/local.
- Both nonfused kernels: 40 registers, 1,920 B shared, zero stack/local.
- Existing accepted Q6 N=1 fused/nonfused instruction encodings are exact.
- Guarded host/device commitment is 2,005,008,384 B, below 2 GiB.

## Timing

Ten independent balanced B/C pairs used 20 warmups and 100 timed full-FFN graph
iterations per process.

| Metric | Result |
|---|---:|
| Baseline median | 2.658982 ms |
| Candidate median | 2.658110 ms |
| Paired median throughput change | +0.614259% |
| Paired IQR | [-1.477448%, +1.759534%] |
| Wins | 6/10 |
| Seeded bootstrap 95% interval | [-1.721576%, +1.898668%] |

The gate required >=+2%, >=8/10 wins, and a wholly positive interval. All three
failed. No repeat, profiler, loader, model conversion, or full-model run was
performed.

## Conclusion

The partial SoA (`ql` plus 82-byte tail) measured +0.769%; this full SoA measured
+0.614%. Both had only 6/10 wins and confidence intervals crossing zero.
Therefore exact-size address-only field rearrangement is closed. A credible next
repack would need a GPU-native AoSoA/tile format plus a new cooperative
lane-to-data mapping, so that several lanes/warps consume aligned vector tiles
rather than retaining the existing one-warp/one-row access pattern.

Evidence is in `correctness/`, `static/candidate/`, and `timing-set1/`.
