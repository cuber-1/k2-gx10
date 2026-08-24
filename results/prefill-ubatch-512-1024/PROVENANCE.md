# Provenance

- Build/source: `build-decode-prefetch-server` from
  `vendor/llama-decode-prefetch-server`; no source file was edited.
- `llama-bench` SHA-256:
  `5e93c3ffd47a456a65b6de48cd26573fdf7d98d74e436c58805cdcfe6e97d95e`.
- `libllama.so.0.0.0` SHA-256:
  `0d342c35039012e6c1529e1b64d406dba5b30669878c7f78d49e11bc9ed915b6`.
- `libggml-cuda.so.0.19.0` SHA-256:
  `4a951282b5d024d3e9673baaed46d797effb493022975faabf49251512dec362`.
- Gate harness SHA-256:
  `b1f0c016c9ef57590000f9a8d75fcba5f705ba3042f64d6f11d82d96e89cf713`.
- Diagnostic harness SHA-256:
  `4a243e5429a346ddb227f12bd71c8ae99f9972098c2db285caa1a92cd91bda4d`.
- The gate harness's `ldd.txt` resolves llama, ggml, CPU, and CUDA only from
  `build-decode-prefetch-server/bin`; `llama-bench-ldd.txt` records the planned
  timing binary's equivalent mapping.
- The cached snapshot and four resolved blob IDs/sizes are recorded in
  `model-shards-provenance.txt`. No model was downloaded.
- Both correctness runs used nonblocking `flock` on
  `/tmp/k2-gx10-gpu.lock`, a finite timeout, an empty Ollama list, and no other
  NVIDIA compute application. Raw checks are under `correctness/set1/` and
  `correctness/determinism/`.
- The deterministic token file contains exactly 2048 signed 32-bit token IDs
  and has SHA-256
  `af8c451c576aa6d5e9b6f417a583d2c26aa3b092fff637c7e951f825a877556c`.
