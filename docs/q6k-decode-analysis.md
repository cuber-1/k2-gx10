# Q6_K decode matvec analysis on GB10

## Current project state (2026-08-19 EDT)

The accepted decode baseline is the DGX-Spark Q6_K N=1 eight-warp mapping plus
distance-one L2 prefetches at offsets 0 and 128. Relative to the eight-warp-only
build, the prefetch change improved the full 73B model's generation throughput
by 10.4636% (3.549480 to 3.920885 tok/s by pooled median), after two reproducible
microbenchmark sets and byte-identical GPU-output checks.

The accepted prefetch gain also persists with Flash Attention, f16 K/V, CUDA
graphs, a fixed 8192-token context, and occupied KV depths through 7168. Two
independent ten-pair full-model campaigns passed every preregistered gate at all
four tested depths; the confirmation measured paired median throughput gains
from 11.2610% to 11.7542%, with 10/10 wins at each depth.

The cooperative Q6 high-bit loader was statically cheaper and correct, but it
failed the timing gate. Decode remains on the accepted Stage-A build. Prefill
`ubatch=1024` was also rejected before timing because its deterministic final
logits exceeded the numerical gate relative to 512. Q6_K/J128 factor-four
unrolling then failed its compile-only gate by quadrupling stack use. CUDA's
Q6-translation-unit register-policy level four compiled to identical SASS and
was also rejected before GPU use. The built-in cuBLAS fallback passed synthetic
numerical checks but was 18.71% slower at N=1024. The exact-arithmetic MMQ
`vsub4` simplification also regressed 2.28%. A final bounded Q8-index
recurrence refactor shortened the target kernel by eight instructions but
worsened its stack and local-memory traffic, so it too was rejected before GPU
execution. The final bounded prefill specialization also failed its spill gate,
and the final decode-only partial unroll increased fused register use by 50%.
Both were rejected before GPU execution. Bounded exact-arithmetic
micro-optimizations are exhausted. A separate q8_0 K-cache experiment then
failed its preregistered numerical gate before timing. Final work is limited to
the packaged, validated DGX Spark decode path. An explicitly approved cooperative
full-FFN megakernel redesign was subsequently implemented and tested, but both
its original and occupancy-restored forms were about 4.3-4.7% slower than the
accepted graph. It was rejected without changing the accepted patch.
The first major repack prototype then split each exact-size Q6_K row group into
contiguous 128-byte `ql` fields and unchanged 82-byte tails. It was byte-exact,
spill-free, and sanitizer-clean, but achieved only +0.769% paired median full-FFN
throughput with 6/10 wins and a confidence interval crossing zero. It too was
rejected without changing the accepted patch. A stronger full-field SoA then
separated `ql`, `qh`, scales, and deltas; it also failed at +0.614%, 6/10 wins,
with a confidence interval crossing zero.

Current assignments are deliberately non-overlapping:

- Experiment agent: completed the isolated long-context validation and is idle.
- Review agent: accepted both long-context campaigns after independent raw-data,
  statistics, provenance, and safety audits.
- Research agent: completed the bounded-candidate exhaustion audit; the
  conditional Nsight dilution fallback was not triggered.
- Manager: final result packaging, project record, and handoff.

## Selected kernels

The original bounded full-model Nsight Systems report is
`profiles/nsys/k2-20260813-174541.nsys-rep`. Its Q6_K matvec kernels account
for about 62% of the explicit eager/capture-time kernel table. That report used
the default graph-level tracing mode, so it did not expose kernels replayed as
CUDA graph nodes.

The two largest specializations are:

| Kernel | Full-profile share | Representative grid | Block | Registers/thread | Static shared memory |
|---|---:|---:|---:|---:|---:|
| `mul_mat_vec_q<Q6_K,2,false,false>` | 29.9% | `(14336,1,1)` | `(32,4,1)` | 56 | 1536 B |
| `mul_mat_vec_q<Q6_K,1,true,false>` | 26.8% | `(28672,1,1)` | `(32,4,1)` | 46 | 768 B |

Both use zero dynamic shared memory and report zero local memory per thread.
The first computes two output columns per launch. The second fuses two Q6_K
projections (FFN up and gate) with SwiGLU for one decode column.

The corrected graph-node-aware capture is
`profiles/nsys/k2-20260817-211725-graph-nodes.nsys-rep`. It recorded 3,052
graph-node kernel rows totaling 654.441 ms. Q6_K MMVQ contributed 642.679 ms,
or 98.20% of graph-node kernel duration. All Q6_K graph-node launches used the
single-column specializations; the old N=2 share was capture-time work rather
than steady single-token replay. This makes N=1 MMVQ the primary decode target.

## Isolated reproduction

The guarded synthetic benchmark now supports the fused decode graph while
retaining the original non-fused and MMQ modes. It loads no model and accepts
no model path.

```bash
cd /home/dvijraicha/k2-gx10

# Single-column non-fused matvec.
./build-microbenchmark/q6k-microbench \
  --execute --columns 1 --warmup 20 --iterations 100

# Two-column non-fused matvec.
./build-microbenchmark/q6k-microbench \
  --execute --columns 2 --warmup 20 --iterations 100

# Dominant fused FFN decode matvec.
./build-microbenchmark/q6k-microbench \
  --execute --columns 1 --fused-swiglu --warmup 20 --iterations 100
```

Five independent runs, each using 20 warmups and 100 timed iterations, gave
these medians of per-run medians:

| Isolated graph | Median |
|---|---:|
| N=1 non-fused | 0.937 ms |
| N=2 non-fused | 0.981 ms |
| N=1 fused SwiGLU | 1.980 ms |

Every run passed the CPU/CUDA numerical gate. The fused NMSE was about
`3.25e-5`, below the `5e-4` limit.

The one-launch Systems report for the fused graph measured 1.994 ms, grid
`(28672,1,1)`, and block `(32,4,1)`. The matching full-model grid had a median
of 1.937 ms. This close agreement validates the isolated shape.

Reports that can be opened directly are:

```bash
nsys-ui /home/dvijraicha/k2-gx10/results/q6k-decode-baseline/n1-fused-swiglu.nsys-rep
nsys-ui /home/dvijraicha/k2-gx10/results/q6k-decode-baseline/n2-nonfused.nsys-rep
```

## Nsight Compute collection

The privileged wrapper now has a decode-specific mode with a fixed kernel
filter, one-launch limit, protected staging, ownership checks, a five-minute
timeout, and a guarded allocation of 1,234,969,888 bytes below the 2 GiB hard
limit.

The completed bounded Stage 2 report can be opened with:

```bash
ncu-ui /home/dvijraicha/k2-gx10/profiles/ncu-microbenchmark/q6k-decode-stage2-bottleneck-analysis.ncu-rep
```

The report contains exactly one
`mul_mat_vec_q<(ggml_type)14,1,true,false>` result and the Speed of Light,
compute, memory, warp-state, source-counter, launch, and occupancy sections.

## Counter diagnosis

The isolated weight payload is 192,675,840 bytes. The non-fused N=1 and N=2
times correspond to roughly 196--206 GB/s of effective weight-payload scan
rate. The fused kernel reads two such matrices and achieves roughly 195 GB/s.
These are useful workload ratios, not hardware-counter bandwidth measurements.

Unlike the MMQ kernel, these matvec kernels have modest register counts and no
dynamic shared-memory allocation. Therefore the previous MMQ strategy of
reducing 255-register pressure is not applicable.

The completed Stage 2 fused-kernel report measured:

| Metric | Value |
|---|---:|
| Duration | 2.007 ms |
| Registers/thread | 46 |
| Theoretical / achieved occupancy | 83.33% / 73.15% |
| Compute / memory throughput | 14.29% / 14.29% |
| Warp cycles per issued instruction | 86.44 |
| Long-scoreboard stall | 54.83 cycles/issue |
| Global-load throttle | 18.40 cycles/issue |
| L1 / L2 hit rate | 75.75% / 7.20% |
| Excessive global sectors | 22,249,472 of 53,473,280 (42%) |

This is a memory-latency and memory-level-parallelism problem. Peak bandwidth
is not saturated, occupancy is already high, and the two dominant stall reasons
are waiting on global data and throttling global loads.

## Accepted Stage-0 eight-warp candidate

The first accepted stage added a Blackwell MMVQ parameter table and changed
only single-column Q6_K from four to eight warps per block. Its historical
standalone artifact is `patches/q6k-blackwell-decode-8-warps.patch`; it is not
the current patch to apply. Other quant types, multi-column paths, older CUDA
architectures, and the MMQ prefill kernel retained their existing selection.

A portability audit found that the current `BLACKWELL_MMA_AVAILABLE` dispatch
also covers sm_120a and future 12.x fallbacks, although performance evidence is
only from GB10/sm_121a. Before upstream application, narrow both the host and
device table selection to `GGML_CUDA_CC_DGX_SPARK` (1210); changing only one
side could make host launch dimensions disagree with the instantiated kernel.
This is a scope correction, not a new performance candidate.

The rejected two-warp candidate had a median of 2.329 ms across five runs,
about 12% slower than the contemporaneous 2.083 ms baseline. It reduced
available memory-level parallelism too far.

The subsequent two-output-rows-per-block candidate was rejected unbenchmarked
at its predeclared compile/resource gate. Fused N=1 rose from 46 to 56
registers/thread and from 2816 to 4608 bytes of static shared memory. With the
accepted 256-thread block, the register step reduces theoretical residency from
five blocks/40 warps (83.33%) to four blocks/32 warps (66.67%). There were no
spills, and the nonfused kernel stayed at 48 registers, but the dominant fused
path failed the gate. Evidence is preserved in
`results/q6k-decode-rpb2/`; this is a resource rejection, not a measured timing
regression.

A DGX-Spark-only `__vsub4` unpack candidate was also rejected after normal
timing. Its arithmetic proof and compilation were successful: fused SASS fell
from 392 to 368 encoded instructions with the same 46 registers and no spills,
nonfused registers fell from 48 to 44, and CPU-reference correctness passed for
N=1 through N=8. Despite that static improvement, ten interleaved fused pairs
measured a paired median change of -0.316%, only 4/10 wins, and a bootstrap 95%
interval of [-1.476%, +0.574%]. Nonfused was a noisy +0.224% with 6/10 wins.
The change therefore failed every predeclared performance gate. Full evidence
is in `results/q6k-decode-vsub4/`.

For the eight-warp candidate, ten alternating baseline/candidate pairs used 20
warmups and 100 timed iterations per side. Candidate order alternated in each
pair to reduce drift:

| Fused graph result | Baseline | Eight warps |
|---|---:|---:|
| Mean of per-run medians | 2.0667 ms | 1.9604 ms |
| Mean paired speedup | - | 5.10% |
| Pair wins | - | 10 / 10 |

All candidate runs passed the CPU/CUDA gate. Fused NMSE was
`3.25271543e-05`, below the `5e-4` threshold. The non-fused N=1 and unchanged
N=2 paths also passed. Reduction ordering changes the last few bits, as
expected, without changing the error class.

Nsight Systems confirmed that the patched specialization launches grid
`(28672,1,1)` and block `(32,8,1)`. Its captured fused launch was 1.801 ms,
versus 1.994 ms in the original one-launch capture. These captures did not use
fixed clocks, so the alternating microbenchmark is the primary kernel result.

## Full-model decode validation

A separate full llama.cpp build applied only the accepted patch. Two blocked
candidate/baseline comparisons used the cached four-shard model and this shape:

```bash
llama-bench -m K2-Think-V2-Q6_K-00001-of-00004.gguf \
  -ngl 99 -p 0 -n 44 -fa off -r 5 --delay 1 -o json
```

| Build | Block 1 | Block 2 | Combined mean |
|---|---:|---:|---:|
| Baseline | 2.9171 tok/s | 2.9140 tok/s | 2.9155 tok/s |
| Eight-warps candidate | 3.0577 tok/s | 3.0459 tok/s | 3.0518 tok/s |

The combined improvement is 4.67%. Frequencies were not fixed, but the result
persisted across both candidate-then-baseline blocked comparisons and is much
larger than the within-block standard deviations. Across all ten samples, the
candidate minimum of 3.02668 tok/s exceeded the baseline maximum of 2.94155
tok/s.

Raw summary rows are in
`results/q6k-decode-nwarps8/microbenchmark-pairs.csv` and
`results/q6k-decode-nwarps8/full-model-blocks.csv`; all 20 full-model timing
samples are in `results/q6k-decode-nwarps8/full-model-samples.csv`. The
candidate one-launch Systems report is
`results/q6k-decode-nwarps8/fused-launch.nsys-rep`.

## Accepted distance-one L2 prefetch

On top of the accepted eight-warp baseline, the next accepted candidate adds
DGX-Spark-only distance-one `prefetch.global.L2` hints for byte offsets 0 and
128 of the next 210-byte Q6_K block. Lane 0 of each warp issues the hints. The
fused path covers both up and gate matrices; the nonfused path covers its one
weight matrix. Bounds, other types, N>=2, small-K, non-Spark CUDA, HIP, and
MUSA paths are unchanged.

Exact fused SASS contains four `CCTL.E.PF2` hints, retains 21 global loads/Q8
reuse and four DP4As, and has 48 registers, 2816 bytes static shared memory,
and no spills. The five-CTA/40-warp theoretical residency therefore remains
83.33%. Direct accepted-baseline/candidate GPU outputs match byte for byte for
N=1 fused, N=1 nonfused, and N=2.

Two independent microbenchmark sets each used ten alternating pairs with 20
warmups and 100 timed iterations. Fused paired medians improved 14.49% and
14.44%, with 10/10 wins in both sets. Nonfused improved 12.13% in both sets,
also 10/10 each. Every bootstrap interval excluded zero.

Two full-model candidate/baseline blocks used the same generation-only
`llama-bench` command as the eight-warp validation. Across ten samples per
build, the accepted baseline median was 3.549480 tok/s and prefetch was
3.920885 tok/s, a 10.4636% increase. Candidate range 3.906080-3.929290 tok/s
did not overlap baseline range 3.537980-3.556080 tok/s. Complete evidence is
in `results/q6k-decode-prefetch-0-128/`.

The consolidated accepted artifact is
`patches/q6k-gb10-decode-final.patch`. It contains both accepted stages and
narrows the host and device special-table selectors to exactly DGX Spark cc
1210. The scope-only selector correction produces a byte-identical sm_121a
MMVQ cubin to the measured Stage-A build; non-1210 targets retain the generic
table and contain no DGX-Spark prefetch hints. Packaging and static-comparison
evidence is in `results/q6k-decode-final-package/`.

### Accepted long-context decode validation

A preregistered full-model experiment isolated the accepted 0/128-byte prefetch
against the eight-warp-only build at occupied KV depths 0, 2048, 4096, and 7168.
Every process used the exact four-shard model, fixed `n_ctx=8192`, Flash
Attention, f16 K/V, CUDA graphs, batch 2048, ubatch 512, and 128 synchronized
single-token decodes. The first complete depth-specific repetition was excluded
as warmup. Each campaign used ten fresh, balanced A/B process pairs, and the
confirmation reversed both build-first scheduling and depth order.

| Depth band | Primary paired median (95% CI) | Confirmation paired median (95% CI) |
|---|---:|---:|
| 0..127 | +11.9985% ([+11.7895%, +12.2111%]) | +11.7542% ([+11.5729%, +11.9707%]) |
| 2048..2175 | +11.6166% ([+11.3071%, +12.1152%]) | +11.5874% ([+11.4104%, +11.7400%]) |
| 4096..4223 | +11.4940% ([+11.2348%, +15.5916%]) | +11.5457% ([+11.3421%, +11.7453%]) |
| 7168..7295 | +10.9621% ([+6.8133%, +11.6010%]) | +11.2610% ([+11.1505%, +11.6203%]) |

All four depths had 10/10 wins and positive build-order strata in both
campaigns. All 40 processes passed exact workload, context, graph, offload, and
runtime-safety checks. In confirmation, the median absolute latency saving at
depth 7168 retained 99.86% of the depth-zero saving, so the preregistered Nsight
Systems dilution fallback was not triggered. Full raw evidence and independent
statistics are in `results/q6k-decode-long-context-20260818/`.

### Rejected third-line prefetch

A separate Stage-B candidate added only an offset-208 hint to the accepted
0/128 helper. It passed the static gate (48 fused registers, 56 nonfused,
zero spills, six fused prefetch sites, and unchanged load/dot structure) and
produced byte-identical GPU outputs for fused N=1, nonfused N=1, and N=2.
Normal interleaved timing was neutral: fused paired median was +0.0612% with
5/10 wins and bootstrap 95% CI [-0.8235%, +1.3176%]; nonfused was +0.5091%
with 6/10 wins and CI [-0.1287%, +1.4018%]. It failed the predeclared 1% and
confidence gates, so there was no repeat, full-model run, or NCU collection.
The 0/128 Stage-A implementation remains the accepted baseline. Evidence is
preserved in `results/q6k-decode-prefetch-0-128-208/`; an independent review
recomputed the same statistics and accepted the rejection.

### Rejected post-prefetch `__vsub4` interaction

One predeclared interaction retest combined the accepted 0/128 prefetch with
the numerically exact DGX-Spark `__vsub4` Q6 unpack. It retained 48 fused
registers, zero spills, all four prefetch sites, 21 fused global loads, Q8 CSE,
and four DP4A instructions; direct GPU outputs matched Stage A byte for byte.
Normal timing improved fused N=1 by only +0.4831% (8/10 wins, bootstrap 95%
CI [+0.1818%, +1.2129%]) and nonfused by +0.5753% (6/10, CI
[-0.8392%, +1.1378%]). The fused result failed the mandatory +1% magnitude
gate and nonfused remained noisy, so the candidate was rejected without a
repeat, full-model test, or NCU run. Evidence is preserved in
`results/q6k-decode-prefetch-vsub4/`.

### Rejected cooperative low-bit loader

An alignment-safe cooperative `ql` loader reconstructed each lane's four
low-bit bytes from one aligned 32-bit load, with a phase-two shuffle and lane-0
16-bit prologue. It passed 107,520 exhaustive single-byte basis cases plus
2,048 dense deterministic cases, compute-sanitizer with zero errors,
byte-identical GPU-output checks, and the
static gate (48 fused registers, 21 loads, four prefetches, four DP4As, zero
spills). Normal timing nevertheless regressed: fused paired median -1.2031%
with 4/10 wins and CI [-2.0833%, +0.3012%], and nonfused -1.4596% with 2/10
wins and CI [-2.8233%, -0.5043%]. The dispatch, shuffle, prologue, and
reconvergence dependencies outweighed the aligned-load benefit. The candidate
was rejected without repeat, full-model testing, or NCU; evidence is in
`results/q6k-decode-prefetch-ql-coop/`.

### Rejected cooperative high-bit loader

An exact-DGX-Spark Q6_K N=1 wrapper replaced each pair of 16-bit `qh` loads
with one aligned 32-bit load and an XOR-8 lane exchange when the 210-byte block
placed `qh` at phase two. Exact fused SASS improved from 21 to 19 global loads
and from 12 to 8 U16 loads while retaining 48 registers, 2816 bytes shared,
zero spills, four Stage-A prefetches, Q8 CSE, and four DP4As. N=2 remained
instruction-identical. The candidate passed 109,568 reconstruction cases,
compute-sanitizer, and byte-identical GPU-output checks.

Normal timing did not justify the extra phase/shuffle machinery. Fused paired
median improved only 0.4543%, with 7/10 wins and a bootstrap interval
[-0.1824%, +1.7565%]. The nonfused paired median was 1.1531% slower, with 3/10
wins and interval [-1.8441%, +0.5051%], so its interval also crosses zero. It
failed all fused promotion gates and was
rejected without repeat, full-model testing, or NCU. Evidence is in
`results/q6k-decode-prefetch-qh-coop/`.

### Rejected post-prefetch `kbx` partial unroll

The exact cc1210 Q6_K N=1/non-small-K path applied factor-two partial unrolling
to the accepted Stage-A `kbx` loop. The experiment was isolated: the nonfused
kernel and all 275 other MMVQ functions were exact encoding/resource matches
to Stage A, while the intended fused symbol contained the unrolled work.

Fused register use rose from 48 to 72 registers per thread while shared memory
remained 2,816 bytes. That violates the compile resource gate and would risk a
material occupancy loss. The candidate was rejected before GPU correctness,
odd-trip tail testing, timing, profiling, or model work. Evidence is in
`results/q6k-decode-kbx-unroll2/`.

### Rejected prefill ubatch 1024

A result-local full-model harness compared `ubatch=512` and 1024 with the same
saved 2048-token sequence, batch 2048, Flash Attention enabled, one final
logits row, and one pinned model load. Both runs succeeded, produced finite
250,112-element logits, and selected argmax token 42. The cross-setting NMSE
was nevertheless 0.00160124428 against a `1e-6` gate, and the maximum logit
difference was 4.181645% against a 1% gate.

A four-context diagnostic proved this was fixed partition-dependent numerical
drift rather than run noise: two 512 contexts were byte-identical, two 1024
contexts were byte-identical, and both cross comparisons exactly reproduced
the failure and the separately loaded set-one hashes. The candidate was
rejected before the planned one-load timing sweep, so no performance claim is
made. Evidence is in `results/prefill-ubatch-512-1024/`; serving stays at batch
2048 and ubatch 512.

### Rejected J128 factor-four partial unroll

The synthetic MMQ harness now accepts bounded `--rows` and `--k` dimensions,
uses checked allocation arithmetic, accounts for the exact Q8 workspace and
Stream-K fixup, and can write non-overwriting result-local GPU outputs. This
made the production-like M=8192,N=512 grid-48/fixup path independently
testable without loading the model.

The exact-DGX-Spark Q6_K/J128 nonfallback candidate compiled the eight-panel
loop into a real two-trip factor-four loop, but failed before GPU execution.
Registers moved 255 to 254 while stack use grew 64 to 256 bytes, static local
loads grew 32 to 208, and stores grew 27 to 160. The apparent IMMA reduction
512 to 256 is static code-size folding; the two loop trips retain the dynamic
math. This violates the mandatory no-worse-spill gate, so no correctness,
timing, profiling, or full-model run was performed. Evidence is in
`results/q6k-prefill-j128-unroll4/`.

### Rejected CUDA register-policy level four

CUDA 13's beta `--register-usage-level=4` option was scoped only to the Q6_K
MMQ translation unit. Compile provenance confirmed Q5_K/Q8_0 and linking were
unchanged; path-normalized PTX was identical. The resulting exact Q6 cubin was
byte-for-byte identical to the default level-five build, and all 64 normalized
function encoding streams matched. J128/nonfallback remained at 255 registers,
a 64-byte stack, 32 static local loads, and 27 stores. With no material spill
improvement, the candidate was rejected before GPU execution. Evidence is in
`results/q6k-prefill-reglevel4/`.

### Rejected forced cuBLAS prefill

The built-in `GGML_CUDA_FORCE_CUBLAS` path was tested at M=28672,K=8192 with
default F16 conversion/GEMM arithmetic. Its corrected conservative commitment
was 1,964,277,760 bytes, below the 2 GiB limit. Synthetic correctness passed at
N=25/512/1024, including strengthened gates limiting NMSE and max-absolute
error to four times the same-shape MMQ baseline.

Five warmed alternating N=1024 pairs triggered the early-rejection rule:
baseline median was 9.022 ms, forced cuBLAS was 10.684 ms, and the paired
median was 18.711656% slower with 0/5 wins. No extended timing, N=512 timing,
profiling, model load, or full-model run followed. Evidence is in
`results/q6k-prefill-force-cublas/`.

### Rejected MMQ non-saturating packed subtract

The exact-DGX-Spark Q6_K MMA loader replaced provably unnecessary signed
saturating byte subtraction with ordinary per-byte subtraction. It removed all
128 saturation-tail permutations from J128/nonfallback and reduced encoded
instructions 8,744 to 8,432 while preserving IMMA, global/shared/local access
counts, 255 registers, and the 64-byte stack. Seven bounded shapes passed CPU
NMSE and produced byte-identical baseline/candidate GPU output.

Normal unprofiled timing nevertheless regressed on the production-like
M=8192,K=8192,N=512 grid-48/fixup shape. Baseline median was 1.9765 ms versus
2.0215 ms candidate; paired median was -2.276759% with 0/10 wins and bootstrap
interval [-3.002807%, -1.710643%]. No controls, repeat, profiling, or model run
followed. Evidence is in `results/q6k-prefill-vsub4/`.

### Rejected terminal Q8-index recurrence CSE

The exact-DGX-Spark Q6_K/J128/nonfallback Stream-K terminal path was changed to
share one Q8 base index between its two half-tile loads. The intended target
was isolated: after normalizing `NO_DEVICE_CODE` `__LINE__` immediates in
unsupported placeholder functions, 63 of 64 Q6 function encodings were identical and only
J128/nonfallback changed. Math, global/shared access counts, barriers, and
branch topology were preserved, and encoded instructions fell from 8,744 to
8,736.

The compile resource gate nevertheless failed. Registers remained 255, stack
usage rose from 64 to 80 bytes, local loads rose from 32 to 34, and local
stores rose from 27 to 29. The candidate was rejected before a full build or
any GPU correctness, timing, profiling, or model work. Evidence is in
`results/q6k-prefill-q8-index-cse/`.

### Rejected direct-grid kernel specialization

The final bounded prefill source candidate specialized only the cc1210
Q6_K/J128/nonfallback shapes for which the existing Stream-K scheduler already
assigns exactly one full tile to each block. It retained the same flat grid,
tile mapping, and arithmetic, but removed the provably unreachable terminal
and fixup body from a new compile-time specialization. The new symbol contained
one 256-IMMA/640-FFMA body, kept 1,024 bytes of shared memory, and did not access
the inherited unused `tmp_fixup` parameter. All mapped generic and fixup
symbols remained exact encoding/resource matches to baseline.

The active body still compiled at 255 registers with a 40-byte stack, 15 local
loads, and 11 local stores, failing the predeclared zero-local-traffic gate.
It was rejected before a full build or GPU work. Evidence is in
`results/q6k-prefill-direct-grid-specialization/`.

### Rejected q8_0 K-cache serving experiment

The accepted source was rebuilt in isolation with
`GGML_CUDA_FA_ALL_QUANTS=ON` because the accepted OFF build does not contain
the mixed q8_0-K/f16-V Flash Attention specialization. The ON build contained
that specialization, while its f16/f16 Flash Attention cubin, SASS, and
resources were byte-identical to the accepted OFF build. A same-token parity
run then produced byte-identical f16 logits across the two builds for a
2048-token prefix plus 128 teacher-forced decode steps.

With K=q8_0 and V=f16, two fresh contexts were internally deterministic and
all 129 greedy argmax choices matched f16. KL and selected-token log-prob drift
were small, but aggregate NMSE was 4.80698e-4 against a 1e-4 gate and normalized
maximum absolute difference was 15.5362% against a 2% gate. The candidate was
therefore rejected before timing or server validation. Evidence is in
`results/k-cache-q8_0-depth2048-allquants-on/`.

## Active overnight queue

The accepted eight-warp plus 0/128 prefetch candidate is the baseline for
subsequent decode work. Implementation and GPU timing remain serialized.

1. Keep the accepted Stage-A decode kernel unchanged. The post-prefetch
   partial-unroll candidate is rejected for raising fused registers from 48 to
   72; no further bounded exact-arithmetic decode candidate survived review.
2. The accepted host/device dispatch has been narrowed to DGX Spark cc 1210 in
   the consolidated final patch. Do not broaden it without target-specific
   evidence.
3. Keep gate-load hoisting deferred unless new post-prefetch profiler evidence
   shows residual gate-specific long-scoreboard stalls.
4. Do not pursue `evict_last`, distance-two hints, or the proposed 224-byte
   bulk prefetch without new profiler evidence. The accepted weights are
   one-use streams, distance one already supplies a full dot interval, and a
   bulk request has view-boundary and alignment complications.
5. Keep `--batch-size 2048 --ubatch-size 512`; ubatch 1024 failed its
   deterministic logits gate and host batch size does not alter physical MMQ
   shapes. Factor-four partial unrolling is rejected for sharply worse spills;
   factor eight is not a candidate because the loop already has eight trips
   and is fully unrolled. Register-usage level four is rejected as a
   compile-time no-op, and forced cuBLAS is rejected as 18.71% slower at
   N=1024. MMQ `vsub4` is rejected as 2.28% slower despite cleaner SASS, and
   terminal Q8-index CSE is rejected for worsening stack/local traffic.
   Direct-grid routing is a no-go because its six-wave lower bound is about
   12.3% worse than the balanced split-tile schedule. The direct-grid kernel
   specialization is also rejected because its surviving active body still
   spilled. Prefill's bounded exact-arithmetic queue is exhausted; GPU-side
   Q6_Kx8 repacking and a true half-J double-buffer pipeline are larger
   redesigns.
6. Keep K/V caches at f16. K-cache q8_0 was deterministic and retained all
   tested greedy choices, but failed the preregistered logit NMSE and maximum
   absolute-difference gates, so no timing or server promotion followed.
7. Exact-size Q6 field-SoA repacking is rejected as timing-neutral: +0.769%
   paired median, 6/10 wins, bootstrap 95% interval [-0.964%, +2.557%]. A future
   full-field variant was also neutral at +0.614%, 6/10, interval [-1.722%,
   +1.899%]. A future repack must change the cooperative lane/tile consumption
   pattern rather than only separating fields and adding address arithmetic.

Serving parameters already supported by evidence should stay fixed: CUDA
graphs enabled, Flash Attention enabled, context 8192, and parallelism 1 for
single-stream decode. Batch and ubatch affect prompt processing rather than the
single-token MMVQ shape. Parallelism 2 is a concurrent-throughput tradeoff, not
a single-stream latency optimization. Reasoning-token budget changes output
length and quality, so it must not be reported as a per-token kernel gain.

Rejected decode ideas now include two warps, fused Q8-load CSE (ptxas already
reuses Q8 registers), and scale leader/broadcast (warp coalescing already
merges the duplicate addresses, so shuffles only add work).

## Applying the patch

The source experiment remains isolated; `/home/dvijraicha/llama.cpp` was not
modified. Apply it explicitly when ready:

```bash
cd /home/dvijraicha/llama.cpp
git apply --check /home/dvijraicha/k2-gx10/patches/q6k-gb10-decode-final.patch
git apply /home/dvijraicha/k2-gx10/patches/q6k-gb10-decode-final.patch
cmake --build build --target llama-server llama-bench --parallel 4
```
