# Provenance

- Workspace: `/home/dvijraicha/k2-gx10`
- Date: 2026-08-19, America/New_York
- Accepted source baseline: `vendor/llama-decode-final`
- Initial isolated candidate: `vendor/llama-decode-megakernel`
- Launch-bounds follow-up: `vendor/llama-decode-megakernel-lb5`
- Baseline build: `build-decode-megakernel-baseline`
- Initial candidate build: `build-decode-megakernel`
- Follow-up build: `build-decode-megakernel-lb5`
- Harness: `src/q6k-ffn-microbench.cpp`
- CUDA: 13.0.88, Release, `compute_121a`/`sm_121a`
- Upstream `/home/dvijraicha/llama.cpp` was not modified.

The initial candidate differs from the accepted isolated source only in:

- `ggml/src/ggml-cuda/mmvq.cu`
- `ggml/src/ggml-cuda/mmvq.cuh`
- `ggml/src/ggml-cuda/ggml-cuda.cu`

The follow-up differs from the initial candidate only in the megakernel launch bound (`1` to `5` minimum blocks/SM).
All unsupported graphs continue through the existing dispatcher.

