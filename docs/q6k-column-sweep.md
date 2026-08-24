# Q6_K column sweep

## Purpose

This sweep varies only the number of activation columns in one isolated GGML operation:

```text
Q6_K weights: [8192, 28672]
F32 input:     [8192, N]
F32 output:    [28672, N]
```

It does not load a GGUF, start `llama-server`, process text, or measure end-to-end tokens per second. The default remains `N=25`, zero warmups, and one timed launch so the existing profiler workflow remains reproducible.

## Commands

Build:

```bash
./scripts/build_q6k_microbench.sh
```

Run one bounded shape:

```bash
./scripts/run_q6k_microbench.sh --execute --columns 100 --warmup 3 --iterations 10
```

Run the thirteen-shape timing and correctness sweep:

```bash
./scripts/sweep_q6k_columns.sh
```

Add one lightweight Nsight Systems trace per shape to verify actual kernel selection:

```bash
Q6K_SWEEP_WARMUP=1 Q6K_SWEEP_ITERATIONS=3 \
  ./scripts/sweep_q6k_columns.sh --with-nsys
```

Accepted benchmark ranges are `columns=1..2048`, `warmup=0..20`, and `iterations=1..100`. The sweep uses `25, 32, 64, 100, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048`. Values above N=1024 remain multiples of the J=128 tile to avoid partial-tile effects. The guarded allocation remains below 2 GiB at every accepted shape; N=2048 is close to the ceiling and no larger standard sweep shape is permitted.

Generate the timing and Nsight Compute dashboards:

```bash
./scripts/plot_q6k_sweep.py \
  --timing-csv results/q6k-column-sweep/20260817-172654/sweep.csv
```

The command writes PNG and SVG dashboards plus derived CSVs under the timing run's `plots/` directory.

## Verified sweep

The extended 30-sample timing run is `results/q6k-column-sweep/20260817-172654/sweep.csv`. The Systems-backed kernel-selection run through N=512 is `results/q6k-column-sweep/20260817-161353/sweep.csv`.

| Columns | Kernel J | Median CUDA time | Columns/ms | NMSE |
|---:|---:|---:|---:|---:|
| 25 | 32 | 1.207 ms | 20.71 | 2.42734665e-05 |
| 32 | 32 | 1.164 ms | 27.49 | 2.43000772e-05 |
| 64 | 64 | 1.363 ms | 46.96 | 2.43962089e-05 |
| 100 | 112 | 1.512 ms | 66.14 | 2.44726724e-05 |
| 128 | 128 | 1.688 ms | 75.83 | 2.44567607e-05 |
| 256 | 128 | 3.325 ms | 76.99 | 2.44532254e-05 |
| 512 | 128 | 4.853 ms | 105.50 | 2.44736786e-05 |
| 768 | 128 | 7.101 ms | 108.15 | 2.44675812e-05 |
| 1024 | 128 | 9.259 ms | 110.60 | 2.44747070e-05 |
| 1280 | 128 | 11.775 ms | 108.70 | 2.44767637e-05 |
| 1536 | 128 | 14.826 ms | 103.60 | 2.44744257e-05 |
| 1792 | 128 | 17.452 ms | 102.68 | 2.44714971e-05 |
| 2048 | 128 | 21.897 ms | 93.53 | 2.44723617e-05 |

All thirteen shapes passed the `5e-4` correctness threshold. Systems confirmed one `mul_mat_q` target launch and the predicted J specialization for each shape through N=512; larger shapes use the J=128 selection path and were timing/correctness tested without Systems tracing.

Column throughput improves from 20.71 columns/ms at N=25 to a measured peak of 110.60 columns/ms at N=1024, about 5.3 times. It remains in a broad 102–109 columns/ms plateau through N=1792, then falls to 93.53 columns/ms at N=2048, about 15% below the peak. Given the timing spread and unfixed clocks, interpret the broad region rather than tiny point-to-point differences. This confirms that N=25 is a short-input latency regime and locates the isolated kernel's approximate throughput regime.

Host-mediated timings have some spread and GPU clocks were not fixed. Use them for regime selection, not tiny optimization claims.

## Representative Nsight Compute profiles

The existing Stage 2 report covers `N=25, J=32`. The guarded wrapper accepts three additional reviewed Stage 2 shapes, including N=1024 in the observed throughput-plateau region:

```bash
sudo ./scripts/profile_ncu_microbenchmark.sh --stage 2 --columns 100
sudo ./scripts/profile_ncu_microbenchmark.sh --stage 2 --columns 512
sudo ./scripts/profile_ncu_microbenchmark.sh --stage 2 --columns 1024
```

These create:

```text
profiles/ncu-microbenchmark/q6k-stage2-bottleneck-analysis-n100.ncu-rep
profiles/ncu-microbenchmark/q6k-stage2-bottleneck-analysis-n512.ncu-rep
profiles/ncu-microbenchmark/q6k-stage2-bottleneck-analysis-n1024.ncu-rep
```

Each command retains the exact kernel filter, one-launch bound, fixed expected allocation, protected staging, five-minute timeout, validation, and no-overwrite publication behavior. Arbitrary privileged column counts are rejected.

Compare the three reports in this order:

1. GPU Speed Of Light throughput.
2. Warp State Statistics, especially long scoreboard and LG throttle.
3. Compute pipeline utilization.
4. Memory Workload Analysis.
5. Occupancy and active warps per SM.
6. Source Counters for excessive global sectors and shared wavefronts.

## Full-model validation

The synthetic sweep identifies kernel regimes cheaply. It cannot show llama.cpp batching, attention/KV-cache costs, or end-to-end token rates. After choosing and implementing a kernel change, validate only a few actual K2 prompt lengths, such as approximately 25, 128, and 512 tokenizer tokens, with a fixed generation budget. Record prompt-processing and generation rates separately.
