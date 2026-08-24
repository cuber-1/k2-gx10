# Q6_K GB10 kernel experiments

## Baseline diagnosis

The representative isolated operation is:

```text
Q6_K weights: [8192, 28672]
F32 input:     [8192, N]
F32 output:    [28672, N]
```

At `N=1024`, llama.cpp selects
`mul_mat_q<(ggml_type)14, (int)128, (bool)0>`. The Stage 2 report is
`profiles/ncu-microbenchmark/q6k-stage2-bottleneck-analysis-n1024.ncu-rep`.

The two resources limiting residency are:

- 255 registers/thread. A 256-thread block consumes essentially the full
  65,536-register SM budget, limiting residency to one block.
- 57.86 KiB dynamic plus 1.02 KiB driver shared memory per block. Under the
  active 65.54 KiB shared-memory configuration this also limits residency to
  one block.

The result is 8 active warps/SM and 16.67% theoretical occupancy (16.78%
achieved). Occupancy is a constraint, not the optimization objective: the
two-block experiment below raised the feasible occupancy and still regressed.

Key NCU measurements at N=1024:

| Metric | Value |
|---|---:|
| Compute (SM) throughput | 43.74% |
| Memory throughput | 33.55% |
| Issue slots busy | 38.36% |
| Tensor INT pipe | 43.74% |
| Warp cycles/issued instruction | 5.62 |
| Math-pipe throttle | 0.962 cycles/issue |
| Wait | 1.065 cycles/issue |
| Long scoreboard | 0.465 cycles/issue |
| Barrier | 0.235 cycles/issue |

Source Counters report 39% excessive global sectors and 8% excessive shared
wavefronts. The global-sector warning is dominated by scattered Q6_K 16-bit
loads. L2 hit rate is 90.45%, and the direct load experiments below show that
the profiler's estimated local opportunity is not an end-to-end prediction.

## Experiment results

Absolute time moves with unfixed clocks. Decisions used correctness gates and,
where noted, interleaved baseline/candidate runs. All source experiments were
made in the ignored `vendor/llama-candidate` copy; upstream
`/home/dvijraicha/llama.cpp` was not modified.

| Candidate | Resource/dispatch result | Timing result | Decision |
|---|---|---|---|
| I=64, J<=64, occupancy target 2 | NVIDIA MMA path failed at runtime | CUDA illegal-memory access | Reject |
| I=128, J<=64, occupancy target 2 | J=64; 128 registers/thread; about 48 KiB dynamic shared memory; two blocks/SM feasible | N512 5.801 ms; N1024 10.496 ms | Reject |
| I=128, J<=64, occupancy target 1 | J=64; 255 registers/thread; one block/SM | N512 5.266 ms; N1024 10.310 ms | Reject |
| Remove duplicate Q6 scale loads when threads exceed I | Same J=128 launch; correctness passed | Neutral to about 1% slower in interleaved 100-iteration runs | Reject |
| Alignment-checked 32-bit Q6 loads | Correct, but retains both load paths and a runtime alignment test | N512 7.016 ms; N1024 13.955 ms | Reject |
| Unconditional 32-bit Q6 loads | Odd 210-byte Q6 blocks are only 2-byte aligned | CUDA misaligned-address error | Reject |
| Pad each Q6_K block from 210 to 212 bytes, then use unconditional 32-bit `ql`/`qh`/scale loads | Every numerical field retains its offset and every block becomes 4-byte aligned; storage grows 0.95%; correctness passed at N=25/512/1024 | Five-run median of run medians: N25 1.169 -> 1.289 ms (+10.3%); N512 4.306 -> 4.462 ms (+3.6%); N1024 8.903 -> 8.954 ms (+0.6%) | Reject |
| Pad each Q6_K block to 212 bytes but retain original paired 16-bit loads | Isolates layout/stride cost from the wider-load instruction; correctness passed | Three-run comparison was +2.9% at N25, noisy/slower at N512, and about 3% faster at N1024; no consistent win | Reject |
| Cap tile at J=112 | J=112; still 255 registers/thread; about 56 KiB dynamic shared memory; grid grows 896 to 1120 blocks at N=512 | N512 4.567 ms; N1024 9.778 ms versus about 4.4/8.9 ms contemporaneous baseline | Reject |
| Disable Stream-K for non-fallback Q6_K | Same I=128/J=128 math and resources; regular multidimensional tiling | N512 6.993 ms; N1024 14.475 ms | Reject |
| One-buffer `cp.async.cg` staging of the Q8 activation tile | Blackwell Q6_K J=128 only; 16-byte global-to-shared copies, L1 bypass, no additional shared memory | Interleaved N512: +4.8% and +11.8%; N1024: -0.1% and +5.3%. The apparent -0.1% is noise and the 30-run gate was +9.5%/+6.0% | Reject |
| Synchronous 128-bit, `L1::no_allocate` Q8 activation loads | Blackwell Q6_K J=128 only; four values held per thread before shared store | Interleaved N512 5.110 -> 6.870 ms (+34.4%); N1024 10.600 -> 13.202 ms (+24.5%) | Reject |
| `L1::no_allocate` on the existing paired 16-bit Q6 `ql`/`qh` loads | Blackwell Q6_K J=128 only; no tile, launch, or shared-memory change | Interleaved N512 5.169 -> 5.505 ms (+6.5%); N1024 10.501 -> 10.810 ms (+2.9%) | Reject |

The J=64 Systems traces are under `results/q6k-candidates/i128-j64-occ1`
and `results/q6k-candidates/i128-j64-occ2`. The J=112 trace is under
`results/q6k-candidates/i128-j112-occ1`. Large binary `.nsys-rep` and SQLite
files are intentionally gitignored.

## Lucebox GB10 review and transfer experiments

The Lucebox megakernel tree was reviewed at commit
`7bea91924969f697a3b28e8c19ce67b89a255f46`. Its build identifies `sm_120`
and `sm_121a` as Blackwell and adds three Blackwell-only translation units:
`kernel_gb10_nvfp4.cu`, `prefill_megakernel.cu`, and `prefill_bw.cu`.

The relevant mechanisms are:

- Its native NVFP4 prefill GEMM is a cuBLASLt call using E2M1 FP4 operands
  and vector UE4M3 scales. This is not an instruction that can consume GGUF
  Q6_K blocks. Transferring it requires converting both weights and
  activations into NVIDIA's FP4 layouts and changes model precision.
- Its custom BF16 prefill GEMM uses a 32x128 output tile and two 64-wide K
  buffers. It issues 16-byte `cp.async.cg` copies for the next A and B chunks,
  commits them, waits one group behind, and performs WMMA on the current
  buffers. This genuine load/MMA overlap is the most transferable algorithmic
  idea.
- With the current BTN=128 constants, that pipeline reserves 56 KiB of dynamic
  shared memory (some nearby comments still describe an older 104 KiB BTN=256
  version). Lucebox opts into the larger per-block allocation and launches a
  cooperative persistent grid, with `grid.sync()` between model phases and
  layers.
- Its persistent NVFP4 decode matvec uses 128-bit
  `ld.global.L1::no_allocate` weight loads, 32-bit packed FP4 loads, warp
  shuffles to share group scales, and fused projection/activation/residual
  work. These techniques work with its contiguous FP4 representation and
  small fixed model, but do not remove Q6_K's split `ql`/`qh` unpacking.
- The custom code does not use a hidden `sm_121a` Q6-like tensor-core
  instruction. The hand-written prefill math is BF16 WMMA; the true Blackwell
  FP4 tensor-core selection is delegated to cuBLASLt.

### Attempt A: one-buffer `cp.async` activation staging

**Change.** For only `mul_mat_q<Q6_K, 128, false>`, the two contiguous Q8
activation-tile copies in each K iteration were changed from one 32-bit load
and shared store per thread to 16-byte `cp.async.cg.shared.global` copies.
J=128 and Stream-K were unchanged, and no second shared buffer was allocated.

**Why it could help GB10.** It imports Lucebox's 16-byte L2-to-shared transfer
mechanism, reduces explicit load/store instructions, and avoids putting a
streamed tile in L1.

**Correctness.** N=25, 512, and 1024 passed. Their NMSE values were
`2.42734665e-05`, `2.44736786e-05`, and `2.4474707e-05`, respectively. N=25
uses J=32 and therefore verifies that the untouched small-N dispatch remains
correct.

**Timing.** A 10-warmup/30-iteration gate measured baseline to candidate as
4.219 -> 4.618 ms at N=512 (+9.5%) and 8.850 -> 9.382 ms at N=1024 (+6.0%).
Interleaved 20-warmup/100-iteration process runs gave:

| N | Pair | Baseline (ms) | Candidate (ms) | Difference |
|---:|---:|---:|---:|---:|
| 512 | 1 | 4.790 | 5.019 | +4.8% |
| 512 | 2 | 4.674 | 5.226 | +11.8% |
| 1024 | 1 | 9.400 | 9.391 | -0.1% |
| 1024 | 2 | 9.666 | 10.177 | +5.3% |

**Decision.** Rejected and reverted. Without a second buffer the copy cannot
overlap the current MMA work. The commit/wait operations add overhead to an
already-coalesced activation copy.

### Attempt B: synchronous 128-bit L1-bypassing activation loads

**Change.** The same J=128 activation staging was changed to Lucebox-style
`ld.global.L1::no_allocate.v4.b32` loads followed by 16-byte shared stores.
This removed async commit/wait while preserving Stream-K and shared-memory
size.

**Why it could help GB10.** It tests whether Lucebox's wide streaming load is
the useful part independently of asynchronous pipelining.

**Correctness.** All three required shapes passed with the same NMSE values as
Attempt A.

**Timing.** Interleaved 20-warmup/100-iteration runs measured 5.110 -> 6.870
ms at N=512 (+34.4%) and 10.600 -> 13.202 ms at N=1024 (+24.5%).

**Decision.** Rejected and reverted. Holding a four-register vector before
the shared store is likely a poor fit for a specialization already compiled at
255 registers/thread. Unlike Lucebox's simpler fixed-shape matvec, this kernel
also keeps a large J=128 MMA accumulator live.

### Attempt C: L1-bypassing existing Q6 loads

**Change.** The existing two 16-bit loads composing each 32-bit `ql` and `qh`
value used `ld.global.L1::no_allocate.u16`. Load width, unpacking, shared
layout, J=128, and Stream-K were unchanged.

**Why it could help GB10.** Q6_K weights are streamed while the activation
workspace is reused. Bypassing L1 for weights could avoid displacing more
valuable activation data without introducing the wide-load temporaries from
Attempt B.

**Correctness.** All three required shapes passed with the same NMSE values as
Attempt A.

**Timing.** Interleaved 20-warmup/100-iteration runs measured 5.169 -> 5.505
ms at N=512 (+6.5%) and 10.501 -> 10.810 ms at N=1024 (+2.9%).

**Decision.** Rejected and reverted. The prior NCU report's 90.45% L2 hit rate
and this regression indicate that retaining normal cache behavior is better
than forcing these scattered Q6 loads away from L1.

After all three rejections, `vendor/llama-candidate/ggml` was restored
byte-for-byte to `/home/dvijraicha/llama.cpp/ggml`, rebuilt, and rechecked at
N=25/512/1024. The original checkout was never modified.

## Conclusions and next direction

1. Raising occupancy is not sufficient. Halving register availability and J
   creates enough additional work and/or lost instruction-level parallelism to
   make the kernel slower.
2. J=128 is the best tested tile for the large-N shapes. J=112 does not reduce
   the compiled register count; J=64 reduces it only when forced by a two-block
   launch bound.
3. Stream-K is valuable even when it launches one block per output tile and
   requires no fixup. Its flat work ordering appears materially better than the
   regular tiling path on GB10.
4. The kernel is a mixed legacy-IMMA/Q6 unpack-and-scale scheduling problem,
   not a simple DRAM-bandwidth problem. The next substantial optimization
   must be a Blackwell-specific Q6 staging/IMMA path that reduces unpack and
   scale instructions or genuinely overlaps staging with MMA. The Lucebox
   experiments show that changing copy instructions without a second buffer is
   insufficient. It should retain J=128 and Stream-K initially so only the
   inner pipeline changes.
5. Merely padding the existing array-of-structures Q6_K layout does not realize
   the global-memory-coalescing gains described by tiled CUDA GEMM work. A
   serious repack experiment should keep the 210-byte GGUF representation on
   disk and create a one-time GPU-side tile or structure-of-arrays layout. That
   layout should group `ql`, `qh`, scales, and deltas so neighboring lanes issue
   contiguous vector-width transactions. The cost of repacking must then be
   amortized separately from steady-state matrix multiplication.
6. A real Lucebox-style double buffer is now beyond a small experiment. The
   current J=128 kernel already uses 57.86 KiB under a roughly 65.54 KiB active
   per-block limit; duplicating its approximately 18 KiB Q8 activation tile
   cannot fit. A rewrite would split K/J into smaller pipeline stages, reserve
   two compact Q8 subtiles, issue the next async stage while the current
   subtile executes IMMA, and change `mmq.cuh`, the Q6 loader, and the MMA
   consumption loop together. A tile/SoA GPU weight representation may also be
   required to make the Q6 producer side copyable in contiguous 16-byte units.
7. Two exact-size field-SoA decode prototypes have now been tested. The first made
   every 128-byte `ql` block cache-line aligned while preserving the 82-byte
   tails and total size. Static resources, byte equality, CPU NMSE, and sanitizer
   all passed, but the full-FFN paired median was only +0.769% (6/10 wins, 95%
   interval crossing zero). A second layout fully separated `ql`, `qh`, scales,
   and deltas and still measured only +0.614% (6/10, interval crossing zero).
   Address-only field separation is therefore closed; any next repack must
   redesign cooperative lane/tile consumption.

Any future candidate must first pass N=25, 512, and 1024 correctness, then beat
an interleaved baseline with at least 10 warmups and 100 timed iterations before
an expensive NCU comparison is collected.

Decode work has moved to `docs/q6k-decode-analysis.md`. The guarded benchmark
reproduces the dominant fused and non-fused Q6_K matvec launches without loading
the model.
