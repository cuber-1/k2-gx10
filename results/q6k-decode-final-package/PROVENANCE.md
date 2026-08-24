# Provenance

- Packaging date: 2026-08-18, America/New_York.
- Workspace: `/home/dvijraicha/k2-gx10`.
- Untouched upstream tree: `/home/dvijraicha/llama.cpp`, commit
  `0b1bad14ff204627636aeb1de22ddcd5acb859d4`.
- Upstream `mmvq.cu` SHA-256:
  `c03a7c063a7dcac18c3dbc3c2434fd5956a5f5522cd2116441701615069d92f1`.
- Accepted Stage-A source:
  `vendor/llama-decode-prefetch-server/ggml/src/ggml-cuda/mmvq.cu`, SHA-256
  `83373db345c594eb006a3a75ef18d23fba705c395eed9a3ae3dd26fe81d17f33`.
- Final isolated source:
  `vendor/llama-decode-final/ggml/src/ggml-cuda/mmvq.cu`, SHA-256
  `670a3e4ca68be6bbbf88225489a58d3e7e7940a5f3f55ecb4e8ddc8bc4d579a3`.
- A recursive comparison of the two isolated checkouts differs only in
  `ggml/src/ggml-cuda/mmvq.cu`.
- Consolidated patch SHA-256:
  `8944a9c7ea9798ddf843d4878a0a9e574b00e51bbc49bf883d57705596bb8d31`.
  Applying it to a temporary copy of the upstream source produced a file
  byte-identical to the final isolated source.

## Build

The final `ggml-cuda` target configured and built successfully (exit 0) in
`results/q6k-decode-final-package/build`. A subsequent verbose verification
also exited 0 and is preserved as `static/build-verification.log`; its final
line reports `[100%] Built target ggml-cuda`.

- Host: aarch64.
- CMake: 3.28.3.
- CUDA compiler: CUDA 13.0, build
  `cuda_13.0.r13.0/compiler.36424714_0`.
- Build type: Release; shared libraries enabled.
- CUDA architecture: 121 (`compute_121a`/`sm_121a`).
- Compression: size; FA on; FA all quants off; graphs on; backend DL off;
  NCCL requested; peer max batch size 128.
- Final `mmvq.cu.o` SHA-256:
  `d51e3ecaacad3fe7ff1982ac824c75be64ee1c7134c20ec8102cd8674ad2639de`.
- Accepted Stage-A `mmvq.cu.o` SHA-256:
  `42d16d4cfee1341ba6b928c961c3f352ebd5be0c37b9aba757a9f04a14b4d8e0`.

The host objects differ as expected because the runtime selector changed. The
embedded sm_121a cubins are byte-identical; see `STATIC-COMPARISON.md`.

No model was loaded, no GPU kernel was launched, and no profiler was run during
final packaging.

