# Q6_K L2 prefetch stage A result

Status: **accepted candidate; reproducible microbenchmark and full-model decode
improvements with direct GPU-output equality.**

## Candidate

For DGX Spark (`sm_121a`) Q6_K MMVQ N=1, rows-per-block 1, each warp's lane
0 issues distance-one `prefetch.global.L2` hints at byte offsets 0 and 128 of
the next Q6_K block. The fused path prefetches both `vx` and runtime-enabled
`vgate`; the nonfused path prefetches `vx`. The next-block access is bounded.
No `+208` prefetch is included.

This source path covers every non-small-K Q6_K N=1 shape, not only K=8192.
The full-model run therefore checks the real graph, including K=8192,
K=28672, and other selected grids.

## Integrated gate and correctness

The exact-symbol resource/SASS gate passed. Fused uses 48 registers, 2816
bytes shared memory, and no local memory or stack. It retains Q8 CSE and four
DP4As, and emits two `CCTL.E.PF2` hints for each of `vx` and runtime-enabled
`vgate`. See `DEVICE_RESOURCE.md`.

CPU-reference correctness passed:

- N=1 nonfused NMSE: 2.46054932e-05;
- N=1 fused SwiGLU NMSE: 3.25271543e-05;
- unchanged N=2 control NMSE: 2.40357088e-05.

Direct candidate-versus-baseline GPU output comparison then passed byte for
byte and by SHA-256:

| case | bytes | shared SHA-256 |
|---|---:|---|
| N=1 fused | 114688 | `75c577fb53daa5fc2e92f6f65c29edb89289930f00489ad646b82c472c3442f4` |
| N=1 nonfused | 114688 | `b40de465b21661b7b222808f9ceff44e95c6add9cf7f099cf65dd2581f9603df` |
| N=2 nonfused | 229376 | `37d99efff61a23d2a23ee6526a1f223a6c85735e6a2262320f264860b301001b` |

The output-capable harness is result-local. Its first failed baseline build
log was preserved; the retry added only the unchanged required
`ggml-common.h` shim. The same executable was run with explicit,
provenance-recorded baseline and candidate library paths.

## Unprofiled microbenchmark timing

Each set contains ten interleaved baseline/candidate pairs with alternating
first build. Every invocation used 20 warmups and 100 timed iterations.
Positive paired percentage means candidate faster. Spread is across the ten
invocation medians.

| set | mode | baseline median | baseline IQR | candidate median | candidate IQR | paired median | paired IQR | wins | bootstrap 95% interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | fused N=1 | 1.9105 ms | 0.01350 ms | 1.6360 ms | 0.01525 ms | +14.4912% | 1.3813 pp | 10/10 | [+13.0915%, +14.8613%] |
| 1 | nonfused N=1 | 0.8900 ms | 0.00850 ms | 0.7800 ms | 0.00350 ms | +12.1264% | 0.6331 pp | 10/10 | [+11.1620%, +12.3594%] |
| 2 | fused N=1 | 1.8965 ms | 0.02750 ms | 1.6285 ms | 0.02450 ms | +14.4433% | 1.0976 pp | 10/10 | [+13.6089%, +14.8756%] |
| 2 | nonfused N=1 | 0.8895 ms | 0.01700 ms | 0.7810 ms | 0.00525 ms | +12.1346% | 1.3744 pp | 10/10 | [+11.0092%, +12.7544%] |

Intervals are deterministic 100,000-resample percentile bootstraps of the ten
paired percentage changes' median. Seed is `0x4b325136`. Percentiles use
linear interpolation at position `(n - 1) * p` for p=0.025 and p=0.975.

Raw logs and per-invocation min/max are in `timing-*-set1/` and
`timing-*-set2/`; paired values are in `paired.csv` and `paired-set2.csv`.

## Full-model decode validation

The clean full-model candidate contains only the accepted 8-warp change plus
this 0/128 prefetch relative to upstream. It differs from the accepted server
baseline in `ggml/src/ggml-cuda/mmvq.cu` only, and that file is byte-identical
to the microbenchmark candidate. Both builds are Release `sm_121a` builds.

The exact command for each block was:

```text
llama-bench -m K2-Think-V2-Q6_K-00001-of-00004.gguf -ngl 99 -p 0 -n 44 -fa off -r 5 --delay 1 -o json
```

Two candidate-then-baseline blocked comparisons (C/B/C/B) preserved all 20
samples:

| block | accepted 8-warp baseline | candidate | throughput change | decode-time reduction |
|---|---:|---:|---:|---:|
| 1 | 3.545019 tok/s | 3.915667 tok/s | +10.4555% | 9.4658% |
| 2 | 3.550017 tok/s | 3.925050 tok/s | +10.5643% | 9.5549% |

Across the ten samples per build, baseline median was 3.549480 tok/s (IQR
0.011518, range 3.537980-3.556080) and candidate median was 3.920885 tok/s
(IQR 0.012728, range 3.906080-3.929290), a pooled-median increase of
10.4636%. The two within-block median gains were 10.4571% and 10.5542%.

Raw JSON, stderr, the 20-row sample table, source/build hashes, linked-library
resolution, configure/build logs, and the exact candidate patch are in
`full-model/`.

## Decision

Accept stage A. It is numerically identical on the explicit GPU comparisons,
passes the integrated instruction/resource gate, reproduces on two fresh
microbenchmark sets, and produces a large, consistent normal unprofiled
full-model decode gain. A separate `+208` stage remains a new experiment and
must not be folded into this accepted result without the same gates.
