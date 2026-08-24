# Focused Nsight Compute analysis: K2-Think-V2 Q6_K on GX10

> Historical note (2026-08-17): the permission block described below was later
> resolved through the guarded privileged microbenchmark wrapper. Current NCU
> results and optimization experiments are in `docs/q6k-kernel-experiments.md`.
> The original text is retained as the record of the earlier full-model attempt.

## Status and bottleneck classification

The focused metric collection is **blocked**, so the low-level bottleneck cannot yet be classified as memory-bandwidth-bound, arithmetic-throughput-bound, occupancy-bound, dependency-bound, or memory-access-inefficient.

Nsight Compute reached the selected kernel but returned `ERR_NVGPUCTRPERM`: the current user cannot access NVIDIA GPU performance counters. Per project safeguards, no permission, kernel module, security, driver, CUDA, or clock setting was changed, and no `sudo` command was used. No `.ncu-rep` was produced.

This is the evidence-backed classification available today:

- Workload-level classification: quantized matrix multiplication dominates GPU execution.
- Kernel-level resource observation: the largest individual kernel has unusually high register allocation and substantial shared memory use.
- Roofline/bottleneck classification: **undetermined because the required counters are unavailable**.

It would be incorrect to label the kernel memory-bound or compute-bound from duration and static resource allocation alone.

## Existing Nsight Systems evidence

The source report is `profiles/nsys/k2-20260813-174541.nsys-rep`; its text summary is `profiles/nsys/k2-20260813-174541-stats.txt`.

| Kernel specialization | GPU kernel time | Instances | Mean duration |
|---|---:|---:|---:|
| `mul_mat_q<(ggml_type)14, 32, false>` | 35.9% | 557 | 684.840 us |
| `mul_mat_vec_q<(ggml_type)14, 2, false, false>` | 29.9% | 557 | 569.599 us |
| `mul_mat_vec_q<(ggml_type)14, 1, true, false>` | 26.8% | 243 | 1,170.383 us |
| `mul_mat_vec_q<(ggml_type)14, 1, false, false>` | 5.3% | 244 | 231.385 us |

These four Q6_K kernels account for 97.9% of recorded GPU kernel time. Type value `14` is `GGML_TYPE_Q6_K`. The single largest specialization is:

```text
void mul_mat_q<(ggml_type)14, (int)32, (bool)0>(...)
```

The Systems database also records 254 registers/thread, a `32 x 8 x 1` block (256 threads), and 44,160 bytes of dynamic shared memory for this specialization. These are static launch/resource facts, not achieved-occupancy measurements.

## Source-code mapping

- Q6_K dispatch: `/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/mmq.cu:44`
- Kernel and Stream-K implementation: `/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:944`
- Tile processing and shared-memory staging: `/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:868`
- Launch and shared-memory sizing: `/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1388`
- Q6_K configuration for J=32: `/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/mmq-config-ampere.cuh:199`
- Q6_K dot product: `/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:650`
- Q6_K template instantiation: `/home/dvijraicha/llama.cpp/ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu:5`

On this backend, non-FP4 Blackwell types fall back to the Ampere MMQ configuration table. The selected Q6_K/J=32/non-fallback entry uses 256 threads, target occupancy 1, tile `I=128`, tile `J=32`, Q6_K shared-memory layout, Stream-K enabled, and no bounds-check fallback. The inner product expands Q6 values and accumulates SIMD integer dot products with scale application. The source contains both DP4A and MMA utility paths; actual Tensor Core utilization requires profiler counters and remains unknown.

## Requested metric analysis

| Requested evidence | Result |
|---|---|
| Representative kernel duration | Systems mean 684.840 us; no focused NCU duration |
| DRAM throughput vs peak | Unavailable: counter permission blocked |
| Compute throughput vs peak | Unavailable: counter permission blocked |
| Tensor Core utilization | Unavailable; source support does not prove runtime utilization |
| Achieved occupancy | Unavailable |
| Theoretical occupancy | NCU unavailable; source target is one block/SM for this config |
| Registers/thread | 254 from Nsight Systems |
| Shared memory/block | 44,160 bytes dynamic from Nsight Systems |
| Dominant warp stalls | Unavailable |
| Memory load efficiency/access patterns | Unavailable |

## Prioritized optimization hypotheses

These are hypotheses, not conclusions. They should be tested only after counter access is available and the focused report establishes the limiting subsystem.

1. **Register-pressure / occupancy hypothesis.** At 254 registers/thread and 256 threads/block, the kernel may be limited to the source-configured one resident block per SM. If `Occupancy` and `WarpStateStats` show low achieved occupancy plus latency/dependency stalls, a smaller `I` tile or otherwise reduced per-thread accumulator footprint may expose more warps and improve latency hiding.
2. **Instruction/dequantization dependency hypothesis.** The Q6_K dot path performs unpacking, repeated integer `dp4a`, scale loads, and dependent accumulation. If compute pipelines are not near peak while `WarpStateStats` shows math-pipe throttle, short scoreboard, or dependency stalls, reordering/unrolling accumulators or using the existing MMA layout more effectively may help.
3. **Shared/global-memory access hypothesis.** Each iteration stages Q6_K and Q8_1 tiles into padded shared memory with four barriers around two dot-product phases. If DRAM/L2 throughput is high or memory sectors show poor utilization, load vectorization/coalescing and Q6_K layout should be investigated; if barrier stalls dominate, double-buffering or fewer synchronization points may help.

## Smallest safe code experiment to test first

After a successful counter capture, the smallest isolated experiment is to add one alternative Q6_K/J=32 configuration with `I=64` instead of `I=128`, leaving quantization math, J, Stream-K behavior, fallback behavior, and all data formats unchanged. The configuration structure explicitly states that these values should affect speed/register pressure/shared-memory use rather than results.

Expected outcome: fewer accumulators and a smaller tile may reduce register/shared-memory demand and potentially allow higher residency or reduce spills. It may also reduce per-block reuse and increase the number of blocks, so performance can regress if the current kernel is bandwidth-bound or launch/work-partition overhead dominates.

Correctness risks: tile-boundary indexing, Stream-K fixup ranges, partial tiles, and expert/MoE indexing could be exposed differently. The experiment must pass existing CUDA tests plus deterministic output/token checks before performance is considered. No such code change has been made in this project.

## Focused command and launch bound

The filter is:

```text
regex:^void mul_mat_q<\(ggml_type\)14, \(int\)32, \(bool\)0>
```

`--launch-skip 32 --launch-count 1` counts only matching kernels, so at most one launch is collected. Model-loading and unrelated generation kernels are excluded. The orchestrated command is in `scripts/profile_ncu.sh`; the exact failed-attempt command and error are preserved in `profiles/ncu/k2-q6k-mmq-focused-q6k-001-report.txt` and `profiles/ncu/k2-q6k-mmq-focused-q6k-001-ncu.log`.

To retry manually on a system where the administrator has already made performance counters available, run:

```bash
cd /home/dvijraicha/k2-gx10
NCU_RUN_ID=focused-q6k-001 ./scripts/profile_ncu.sh
```

Do not weaken Linux security settings solely for this experiment. The profiler report cannot be generated under the current permission policy.
