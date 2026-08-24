# Q6_K FFN cooperative megakernel result

Status: **REJECTED**. The accepted DGX Spark eight-warp plus 0/128-byte L2-prefetch decode path remains unchanged.

## What was tested

An isolated exact-shape CUDA prototype replaced the four-node K2 decode FFN graph with one cooperative launch:

1. F32 hidden-state quantization to Q8_1 (`8192` values);
2. Q6_K up and gate projections plus SwiGLU (`8192 -> 28672`);
3. Q8_1 intermediate quantization (`28672` values);
4. Q6_K down projection (`28672 -> 8192`).

The kernel retained the accepted eight-warp row reduction and distance-one Q6_K L2 prefetches at offsets 0 and 128.
Three cooperative grid synchronizations separated the dependent phases. Dispatch was restricted to CUDA cc 1210,
Q6_K, N=1, SwiGLU, and the exact K2 dimensions.

## Correctness and safety

- Baseline, initial megakernel, and launch-bounds follow-up GPU outputs were byte-identical. All three output files have
  SHA-256 `d5868362d8792632ee78479c217b8bbaa4dc5697691250db9ac6faf8b985e26b`.
- CPU comparison passed: NMSE `2.32741002e-4` against the `5e-4` gate.
- Compute Sanitizer reported zero errors and zero leaked bytes.
- Nsight Systems observed exactly two megakernel launches for one warmup plus one measured graph computation and no
  standalone Q8 quantization or MMVQ launches on the candidate path.
- The guarded allocation was `1,426,980,864` bytes (`1360.875 MiB`), below the 2 GiB limit.
- All 276 pre-existing MMVQ function encoding streams remained exact matches to the accepted final cubin; the megakernel
  was the sole added device symbol.

## Static resources

| Variant | Registers/thread | Shared | Stack/local | Resident grid |
|---|---:|---:|---:|---:|
| Initial megakernel | 64 | 2816 B | 0/0 | 192 CTAs (4/SM) |
| `__launch_bounds__(256, 5)` | 48 | 2816 B | 0/0 | 240 CTAs (5/SM by resource bound) |

The follow-up restored the intended five-CTA residency without spills. Its failure therefore is not attributable only to
the initial 64-register allocation.

## Unprofiled timing

Each set used ten balanced process-level A/B pairs, 20 warmups, 100 measured iterations, and the accepted graph as the
baseline. Speedup is `100 * (baseline_ms / candidate_ms - 1)`.

| Candidate | Baseline median | Candidate median | Paired median | Wins | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Initial 64-register megakernel | 2.523750 ms | 2.653542 ms | **-4.3344%** | 0/10 | [-5.7589%, -3.9834%] |
| 48-register/five-CTA follow-up | 2.538398 ms | 2.658358 ms | **-4.6900%** | 0/10 | [-5.5515%, -3.3445%] |

Both variants fail the >=2%, >=8/10-wins, positive-CI promotion gate and are reproducibly slower.

## Interpretation

The accepted baseline's standalone Q8 quantizers account for only about 0.2% of the profiled FFN kernel time. The two
Q6_K projections dominate. Removing three launches and keeping their small intermediates inside one launch does not
remove the Q6 weight scan. Cooperative grid barriers and the persistent row schedule cost more than the launch overhead
saved, even after restoring five-CTA residency.

Do not integrate this prototype into the model or the accepted patch. A future successful megakernel would need a more
fundamental dataflow/layout change (for example persistent multi-layer fusion or Q6 repacking), not merely wrapping the
existing per-row arithmetic in a cooperative kernel.

## Evidence index

- Frozen hypothesis: `HYPOTHESIS.md`
- Initial timing: `timing-set1/SUMMARY.txt`, `timing-set1/paired.csv`
- Occupancy follow-up: `timing-lb5-set1/SUMMARY.txt`, `timing-lb5-set1/paired.csv`
- Correctness/safety: `outputs/`, `smoke/candidate-cpu.stdout`, `smoke/sanitizer.stdout`
- Launch identity: `smoke/candidate-identity.nsys-rep`, `smoke/candidate-kernels.csv`
- Static cubins/SASS/resources: `static/`

