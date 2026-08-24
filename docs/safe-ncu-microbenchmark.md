# Safe Q6_K Nsight Compute microbenchmark

## Purpose and safety boundary

This replaces the disabled full-model Nsight Compute workflow. The benchmark constructs one synthetic GGML `MUL_MAT`; it does not link llama model-loading code, accept model paths, parse GGUF, start `llama-server`, or contact Hugging Face. Execution is opt-in and guarded below 2 GiB. Root execution is refused except when the fixed privileged profiler wrapper explicitly supplies `--allow-root-profile`; all runs retain a finite timeout and scoped process-group cleanup.

Do not execute `scripts/profile_ncu.full-model.disabled.sh`. It is preserved only as historical documentation and has no executable bit. `scripts/profile_ncu.sh` is now an unconditional refusal wrapper.

## Evidence from the original capture

The source report is `profiles/nsys/k2-20260813-174541.nsys-rep`, with its SQLite export beside it. The exact kernel appears 557 times:

| Grid | Block | Dynamic shared memory | Registers/thread | Count | Mean duration |
|---|---|---:|---:|---:|---:|
| `(48,1,1)` | `(32,8,1)` | 44,160 B | 254 | 399 | 440.646 us |
| `(224,1,1)` | `(32,8,1)` | 44,160 B | 254 | 158 | 1,301.508 us |

The full demangled name is:

```text
void mul_mat_q<(ggml_type)14, (int)32, (bool)0>(const char *, const int *, const int *, const int *, float *, float *, const float *, uint3, int, int, int, int, int, uint3, uint3, int, int, int, uint3, uint3, int, int, int, uint3)
```

The saved request used a 25-token prompt. The 12 MB metadata-only split header reports `embedding_length=8192`, `feed_forward_length=28672`, 80 blocks, and 723 tensors. No tensor payload was loaded to obtain those values.

## Source mapping and specialization

The dispatch path is:

1. `ggml/src/ggml-cuda/ggml-cuda.cu`: F32 input/output and Q6_K weights are eligible for MMQ.
2. `ggml/src/ggml-cuda/mmq.cu`: `GGML_TYPE_Q6_K` dispatches to `mul_mat_q_case<GGML_TYPE_Q6_K>`.
3. `ggml/src/ggml-cuda/mmq.cuh`: rows divisible by 128 select `fallback=false`; `mul_mat_q_switch_J` chooses the supported J producing the fewest column tiles; `launch_mul_mat_q` computes the launch.
4. `ggml/src/ggml-cuda/mmq-config-ampere.cuh`: Q6_K/J=32/non-fallback uses 256 threads, occupancy target 1, I=128, Q6_K shared-memory layout, and Stream-K.
5. `ggml/src/ggml-cuda/mmq.cuh`: the `mul_mat_q` kernel and Stream-K tile processing.
6. `ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu`: Q6_K template instantiation.

GB10 has Blackwell compute capability, but llama.cpp's Blackwell MMA table is for native FP4. Q6_K therefore selects the non-FP4 Ampere MMQ configuration compiled for this device.

## Existing benchmark decision

`tests/test-backend-ops.cpp` is the closest operation-level correctness harness: it can filter to `MUL_MAT` and already compares CUDA with CPU. Its matrix cases are compiled-in, however, and it has no CLI for one arbitrary K2 shape; the exact `8192 x 28672 x 25` case is absent. Extending that large suite would still build and enumerate unrelated tests. The standalone program therefore reuses the same public GGML graph/backend APIs and the same `5e-4` `test_mul_mat` NMSE criterion without invoking a broad suite. `llama-bench` and `llama-server` are not used.

## Representative operation

GGML stores the operation as `C^T = A * B^T`:

```text
src0 Q6_K weight: ne=[8192,28672,1,1] nb=[210,6720,192675840,192675840]
src1 F32 input:   ne=[8192,25,1,1]    nb=[4,32768,819200,819200]
dst  F32 output:  ne=[28672,25,1,1]   nb=[4,114688,2867200,2867200]
```

For 25 input columns, J=8 needs four tiles, J=16 needs two, J=24 still needs two, and J=32 needs one, so J=32 wins. The 28,672 output rows are divisible by I=128, so `fallback=false`. There are 224 row tiles and one column tile. Stream-K sees 224 tiles over 48 SMs at 93% five-wave efficiency and uses the direct 224-block launch, reproducing captured `grid=(224,1,1)`, `block=(32,8,1)`.

## Allocation calculation

Q6_K uses one 210-byte block per 256 values.

| Allocation or enforced allowance | Bytes | MiB |
|---|---:|---:|
| GPU tensor buffer | 196,362,240 | 187.266 |
| CPU reference tensor buffer | 196,362,240 | 187.266 |
| Host Q6_K staging | 192,675,840 | 183.750 |
| Host F32 input | 819,200 | 0.781 |
| Host CPU and GPU outputs | 5,734,400 | 5.469 |
| Initialization scratch allowance | 32,768 | 0.031 |
| CUDA Q8_1 MMQ workspace | 231,552 | 0.221 |
| Stream-K fixup | 0 | 0 |
| GGML metadata allowance | 2,097,152 | 2.000 |
| Unclassified backend reserve | 268,435,456 | 256.000 |
| **Guarded total** | **862,750,848** | **822.783** |
| **Hard refusal limit** | **2,147,483,648** | **2,048.000** |

The program prints this entire table before initializing a backend. Default mode is dry-run. It refuses execution if the guarded total reaches the hard limit.

## Build and run

The CMake build is isolated in `build-microbenchmark/`; it does not modify `/home/dvijraicha/llama.cpp/build`.

```bash
./scripts/build_q6k_microbench.sh
./scripts/run_q6k_microbench.sh
./scripts/run_q6k_microbench.sh --execute
```

The CPU result is the established optimized GGML CPU path. The CUDA result passes when its normalized mean squared error is at most `5e-4`, matching llama.cpp's upstream `test_mul_mat` criterion. The runner limits the complete execution to 180 seconds.

## Lightweight Nsight Systems verification

Nsight Systems does not replay kernels or collect privileged performance counters. A single bounded capture can verify the specialization:

```bash
mkdir -p profiles/nsys-microbenchmark
timeout --signal=TERM --kill-after=5s 180s \
  nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output profiles/nsys-microbenchmark/q6k-one-launch \
  ./build-microbenchmark/q6k-microbench --execute
nsys stats --report cuda_gpu_kern_sum \
  profiles/nsys-microbenchmark/q6k-one-launch.nsys-rep
```

## Verified result

The isolated build and default dry-run passed. One unprofiled execution and one lightweight Nsight Systems execution both passed correctness with NMSE `2.42734665e-05` against the upstream `5e-4` limit. The verification report is `profiles/nsys-microbenchmark/q6k-one-launch.nsys-rep`. It contains exactly one target launch: `mul_mat_q<(ggml_type)14, (int)32, (bool)0>`, grid `(224,1,1)`, block `(32,8,1)`, 254 registers/thread, 44,160 bytes dynamic shared memory, and duration 1.148320 ms. The target accounted for 99.5% of traced GPU kernel time; the only other kernel was the required F32-to-Q8_1 activation quantization.

## Privileged Nsight Compute Stage 1 — prepared, not executed

The earlier unprivileged proposal was blocked by `ERR_NVGPUCTRPERM`. The explicitly authorized one-time privileged command, root-override boundary, exact kernel filter, timeout, validation, cleanup, and ownership controls are documented in `docs/staged-ncu-microbenchmark.md`. Only `LaunchStats` and `Occupancy` are enabled.

