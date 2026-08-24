# Final inference-optimization results

This project optimized K2-Think-V2 Q6_K inference in llama.cpp on an NVIDIA
GB10 / DGX Spark (`sm_121a`, CUDA compute capability 12.1). The model payload
was 55.43 GiB and all model layers were offloaded to the GPU.

## Accepted production change

The final patch is
[`patches/q6k-gb10-decode-final.patch`](../patches/q6k-gb10-decode-final.patch).
It changes only the Q6_K, single-token CUDA matrix-vector (`MMVQ`) path on exact
cc 1210:

1. use eight warps instead of four for `N=1`; and
2. prefetch the next Q6_K weight blocks into L2 at byte offsets 0 and 128.

The patch does not change prompt prefill, attention, the transformer KV cache,
sampling, non-Q6 formats, or other GPU architectures.

### Measured decode results

| Comparison | Workload | Throughput result | Notes |
|---|---|---:|---|
| 4 warps -> 8 warps | Full 73B model decode | **+4.67%** | Separate earlier validation |
| 4 warps -> 8 warps | Isolated fused Q6_K kernel | **+5.10% mean paired** | Kernel validation |
| 8 warps -> 8 warps + L2 prefetch | Full 73B model decode | **+10.4636%** | 3.549480 -> 3.920885 tok/s |
| 8 warps -> 8 warps + L2 prefetch | Fused Q6_K microbenchmark | **+14.49%, +14.44%** | 10/10 wins in both repeats |
| 8 warps -> 8 warps + L2 prefetch | Nonfused Q6_K microbenchmark | **+12.13%, +12.13%** | 10/10 wins in both repeats |

The +10.4636% result is an incremental prefetch result against the eight-warp
baseline, not a same-campaign untouched-upstream-to-final measurement. The two
separately measured improvements would compose to approximately +15.6% if they
were perfectly multiplicative, but this project does **not** present that as a
directly measured end-to-end result.

The +10.4636% throughput gain corresponds to about 9.47% less steady decode
time per generated token. Prompt processing is unaffected, so whole-request
speedup depends on the prompt/output-length mix.

### Long-context confirmation

The L2-prefetch contribution persisted with fixed `n_ctx=8192`, Flash
Attention on, f16 K/V, CUDA graphs, and 128 measured generated tokens. Each
depth used ten independent balanced A/B process pairs:

| Occupied KV band | Median throughput gain | 95% bootstrap CI | Wins |
|---|---:|---:|---:|
| 0..127 | **+11.7542%** | [+11.5729%, +11.9707%] | 10/10 |
| 2048..2175 | **+11.5874%** | [+11.4104%, +11.7400%] | 10/10 |
| 4096..4223 | **+11.5457%** | [+11.3421%, +11.7453%] | 10/10 |
| 7168..7295 | **+11.2610%** | [+11.1505%, +11.6203%] | 10/10 |

See
[`results/q6k-decode-long-context-20260818/RESULT.md`](../results/q6k-decode-long-context-20260818/RESULT.md)
for the complete protocol and independent confirmation.

## Accepted operating configuration

- CUDA graphs: enabled. Disabling them reduced generation performance by
  approximately 4.56%.
- Flash Attention: enabled. Disabling it reduced prefill throughput by about
  7.21% at 512 tokens and 16.39% at 2048 tokens; generation also regressed.
- Context: 8192.
- Batch / microbatch: 2048 / 512.
- Parallel slots: 1 for the measured single-stream latency configuration.
- KV cache: f16 K and f16 V.

## Prefill outcome

No prefill optimization was accepted.

Notable tested candidates:

| Candidate | Result | Decision |
|---|---:|---|
| `ubatch=1024` | Deterministic NMSE 0.001601 and 4.18% normalized max-logit delta | Rejected before timing |
| Forced cuBLAS at N=1024 | **18.71% slower**, 0/5 wins | Rejected |
| MMQ non-saturating packed subtract | **2.28% slower**, 0/10 wins | Rejected |
| J128 partial unroll | Stack 64 -> 256 B; local loads/stores sharply increased | Rejected before GPU |
| Register-usage level 4 | Identical machine code | Rejected as a no-op |
| Direct-grid specialization | Retained 40 B stack and local traffic | Rejected before GPU |

The prefill MMQ kernel was already at 255 registers/thread with spill traffic,
so apparently simpler source transformations often caused worse register and
local-memory behavior.

## Other system and kernel experiments

| Candidate | Quantitative result | Decision |
|---|---:|---|
| Cooperative full-FFN megakernel | **4.33-4.69% slower**, 0/10 wins | Rejected |
| Exact-size Q6 field-SoA repack | +0.769%, 6/10, CI crossed zero | Rejected |
| Full-field Q6 SoA repack | +0.614%, 6/10, CI crossed zero | Rejected |
| Additional L2 prefetch at +208 | +0.061% fused, CI crossed zero | Rejected |
| Cooperative QL loader | 1.20% slower fused | Rejected |
| Cooperative QH loader | +0.454% fused; nonfused 1.15% slower | Rejected |
| Decode packed `vsub4` | No qualifying fused gain | Rejected |
| Q8_0 K-cache, f16 V | All argmaxes matched, but NMSE and max-logit gates failed | Rejected before timing |

The megakernel did not remove the dominant roughly 578 MB Q6 weight scan per
FFN. It saved only small quantizer/launch overhead while adding three
cooperative grid barriers and forcing several phases with different natural
grid shapes through one persistent schedule.

## Correctness and portability

- Accepted prefetch outputs were byte-identical for fused N=1, nonfused N=1,
  and the unchanged N=2 control.
- The packaged `sm_121a` MMVQ cubin was byte-identical to the measured accepted
  cubin across all 276 logical symbols and resources.
- An actual `sm_120a` control selected the generic path and emitted no L2
  prefetch instructions.
- The original `/home/dvijraicha/llama.cpp` working tree was not modified;
  changes were developed in isolated copies and delivered as one patch.

## Reproduce and inspect

- Consolidated patch: [`patches/q6k-gb10-decode-final.patch`](../patches/q6k-gb10-decode-final.patch)
- Full decode notebook/report: [`docs/q6k-decode-analysis.md`](q6k-decode-analysis.md)
- Final package proof: [`results/q6k-decode-final-package/RESULT.md`](../results/q6k-decode-final-package/RESULT.md)
- Long-context proof: [`results/q6k-decode-long-context-20260818/RESULT.md`](../results/q6k-decode-long-context-20260818/RESULT.md)
- Megakernel report: [`results/q6k-decode-ffn-megakernel/RESULT.md`](../results/q6k-decode-ffn-megakernel/RESULT.md)
- Prefill experiment log: [`docs/q6k-kernel-experiments.md`](q6k-kernel-experiments.md)

The GGUF model, local llama.cpp source copies, compiled binaries, profiler
captures, and multi-gigabyte raw build artifacts are intentionally excluded
from Git. They are not required to inspect or apply the patch.
