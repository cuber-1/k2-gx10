# Full-field Q6 SoA provenance

- Accepted source: `vendor/llama-decode-final`
- Candidate source: `vendor/llama-decode-q6-full-soa`
- Upstream `/home/dvijraicha/llama.cpp`: untouched
- Candidate `mmvq.cu` SHA-256:
  `d1aa51cd59b5e44190e39e94afec9c78bf59f39c65d25562aeadeb9c669398d3`
- Harness SHA-256:
  `8512a3117cb27ae4531522ecab53199602eca2aac3ee73977577c932b2fc22d8`
- Baseline binary/library:
  `5a23427de212aa315c3a99f52b20c2c6a9a0f4b9eb8cef1a9fb03b00ba8cdc49` /
  `0ab764821b958fe973a6c806392690a293b91e3f30b9fd4a078f31baced29093`
- Candidate binary/library:
  `c2ba0c0ed502bfe08f5f9f7843653743f6aefdf38d95adaa544e9d43ad389a23` /
  `aaa4735b3092fda4440d5c6bf8fed75d4b30a305d7dcba432e765fd8b7a409c3`
- Timing runner SHA-256:
  `0d2b05eea6eaf2406768b0cdd13d084e98879b3f80f4de677acbe184362babad`
- Analyzer SHA-256:
  `78056a2116d37eb840f5c8af5f37abff3ab5b4910018c14d9aeece3c34f11505`

Release sm_121a builds were linked to separate candidate/baseline library
stacks, recorded in `timing-set1/*.ldd`. Timing was unprofiled under the shared
GPU lock with alternating B/C order, no compute peer, and no loaded Ollama model.
