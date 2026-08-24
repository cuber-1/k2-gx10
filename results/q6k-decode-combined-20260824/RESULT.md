# Direct combined decode result

Status: **confirmed on the full 73B model.**

This experiment directly compared the untouched upstream Q6_K four-warp
decode kernel against the final combined eight-warp plus 0/128-byte L2-prefetch
kernel. It replaces the previous combined estimate with a same-campaign A/B
measurement.

## Result

| Metric | Untouched 4 warps | 8 warps + L2 prefetch | Change |
|---|---:|---:|---:|
| Pooled median generation throughput | 3.238285 tok/s | 3.804630 tok/s | **+17.4890%** |
| Steady decode time per token | 0.308805 s | 0.262838 s | **-14.8857%** |

The combined build won all **5/5** independent process pairs. The median of
the five paired gains was **+18.0678%**, with a range of +13.6657% to
+21.7233%.

## Protocol

- Full K2-Think-V2 73B Q6_K four-shard model (55.43 GiB payload).
- NVIDIA GB10 / DGX Spark, all 99 model layers offloaded.
- Fixed `n_ctx=8192`, depth 0, Flash Attention on, CUDA graphs on, f16 K/V.
- Batch / microbatch 2048 / 512, 20 CPU threads.
- 128 measured generated tokens per repetition.
- Five fresh process-level A/B pairs, two measured repetitions per process
  after llama-bench's built-in warm-up: ten samples per build.
- Build-first order alternated by pair: baseline/final, final/baseline, and so
  on.
- A preload harness verified the effective context and deterministic random
  work signature for both sides of every pair.

The candidate binary is the previously validated Stage-A full-model build. On
GB10 its selected MMVQ cubin is byte-identical to the packaged final patch; the
later packaging change only narrowed the architecture selector.

## Reproduce and inspect

Run `scripts/run_combined_decode_ab.sh` from the repository root. The runner
refuses to overwrite an existing result directory; set
`COMBINED_DECODE_RESULT_DIR` for another campaign.

- Compact result: `summary-direct.json`
- Per-process comparisons: `paired-invocations-direct.csv`
- All measured samples: `paired-invocations-samples.csv`
- Analyzer: `scripts/analyze_combined_decode_ab.py`

Raw JSON, stderr, binary linkage, hardware state, and hashes are retained in
the local ignored result directory. The compact files above are committed for
GitHub review.
