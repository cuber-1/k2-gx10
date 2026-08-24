# Full-model provenance

- Accepted baseline source: `vendor/llama-decode-server`
- Accepted baseline build: `build-decode-server`
- Isolated candidate source: `vendor/llama-decode-prefetch-server`
- Isolated candidate build: `build-decode-prefetch-server`
- Candidate source differs from accepted baseline only in
  `ggml/src/ggml-cuda/mmvq.cu`.
- Candidate `mmvq.cu` SHA-256 is
  `83373db345c594eb006a3a75ef18d23fba705c395eed9a3ae3dd26fe81d17f33`,
  identical to the validated microbenchmark candidate.
- Both builds use Release, CUDA enabled, `CMAKE_CUDA_ARCHITECTURES=121`, which
  configures `compute_121a`/`sm_121a`; CUDA graphs and flash-attention support
  remain enabled at build time.
- Runtime benchmark explicitly uses `-fa off`.
- Baseline `llama-bench` SHA-256:
  `30883b4a4c41895487567ee8c956b93316ffa67e973eca58cdffae3992056606`.
- Candidate `llama-bench` SHA-256:
  `5e93c3ffd47a456a65b6de48cd26573fdf7d98d74e436c58805cdcfe6e97d95e`.
- Baseline `libggml-cuda.so` SHA-256:
  `0745d62b2be4c7375a1b40f3bc32ded18313052a2ec27f7ea3deea3a7bd30b97`.
- Candidate `libggml-cuda.so` SHA-256:
  `4a951282b5d024d3e9673baaed46d797effb493022975faabf49251512dec362`.
- `ldd-baseline.txt` and `ldd-candidate.txt` show each executable resolves its
  own build's ggml libraries.
- GPU exclusivity was checked before every block. `ollama ps` was empty and no
  active benchmark, profiler, llama server, or CUDA compute process was found.
- No Nsight Compute run was performed.
