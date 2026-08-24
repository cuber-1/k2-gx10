# Device, build, and dispatch record

Status: complete; the candidate was rejected by the warmed early timing gate.

## Isolation and provenance

- Baseline: frozen `build-prefill-j128-baseline`, harness SHA256
  `1378536c577aac6d8d31d0cd63e2b676480052982dac4d6274c0e32923627bbb`.
- Candidate: fresh `build-prefill-force-cublas` from the byte-identical isolated
  source copy `vendor/llama-prefill-force-cublas`.
- `source-tree-diff.txt` is empty and the normalized source manifests match.
  No source file was edited.
- After normalizing source/build paths, the only compile-command difference is
  the expected `-DGGML_CUDA_FORCE_CUBLAS`; cache isolation is
  `GGML_CUDA_FORCE_CUBLAS=ON` versus baseline `OFF`, with
  `GGML_CUDA_FORCE_MMQ=OFF` in both builds.
- `build/ldd.txt`, `build/readelf-dynamic.txt`, and
  `build/artifact-sha256.txt` record the exact candidate libraries and RUNPATH.

## Runtime dispatch

- The result-local feature probe reports `FORCE_CUBLAS=1` only for the
  candidate. Its compiler/link invocation, explicit library paths, ldd, and
  RUNPATH are recorded in `FEATURE_PROBE_PROVENANCE.md`.
- Exact-library host disassembly shows candidate
  `ggml_cuda_should_use_mmq(...)` lowers to `mov w0, #0; ret`; the baseline
  contains the normal selector. The candidate library uniquely contains the
  `FORCE_CUBLAS` feature string.
- Source-chain evidence records the resulting dequantize/convert/cuBLAS path.
  This made Nsight Systems unnecessary; no Nsight Systems or Nsight Compute
  profiling was run.
- The bounded runner explicitly unsets
  `GGML_CUDA_CUBLAS_COMPUTE_TYPE`. `environment-ggml-cuda.txt` records that no
  parent `GGML_CUDA_*` setting affected the run, so the tested arithmetic is
  the default F16 forced-cuBLAS path.

## Bounded resource gate

The maximum authorized M=28672, K=8192, N=1024 case is conservatively bounded
at 1,964,277,760 bytes (1,873.281 MiB), below the 2 GiB hard limit. The bound is
the frozen harness total 1,428,473,856 B, minus unused MMQ workspace 9,455,616
B, plus 448 MiB dequantized weights, 16 MiB F16 activation, and 56 MiB F16
output temporaries. `run-bounded.sh` admits only M=28672, K=8192 and
N in {25, 512, 1024}, checks exclusivity, and uses the isolated build paths.

No register, stack, or shared-memory comparison is applicable because this
candidate changes host dispatch from the custom Q6_K MMQ kernel family to the
built-in conversion plus cuBLAS path.
