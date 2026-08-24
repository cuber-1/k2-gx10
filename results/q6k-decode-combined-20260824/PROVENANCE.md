# Provenance

- Run date: 2026-08-24, America/New_York.
- Baseline source: untouched `/home/dvijraicha/llama.cpp` at commit
  `0b1bad14ff204627636aeb1de22ddcd5acb859d4`.
- Baseline `mmvq.cu` SHA-256:
  `c03a7c063a7dcac18c3dbc3c2434fd5956a5f5522cd2116441701615069d92f1`.
- Final measured source: `vendor/llama-decode-prefetch-server`.
- Final measured `mmvq.cu` SHA-256:
  `83373db345c594eb006a3a75ef18d23fba705c395eed9a3ae3dd26fe81d17f33`.
- Baseline `llama-bench` SHA-256:
  `a44163dbd117d419e0b100fa1c64684de75f2d196fd3db541703a1c64b235019`.
- Final `llama-bench` SHA-256:
  `5e93c3ffd47a456a65b6de48cd26573fdf7d98d74e436c58805cdcfe6e97d95e`.
- Both builds were Release, CUDA-enabled, Flash-Attention-enabled,
  CUDA-graphs-enabled builds targeting GB10 `sm_121a`.
- The fixed-context harness SHA-256 is recorded locally with both binary
  hashes in `provenance/artifact-sha256.txt`.
