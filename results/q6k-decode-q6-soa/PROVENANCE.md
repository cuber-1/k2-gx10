# Q6 field-SoA provenance

- Accepted source: `/home/dvijraicha/k2-gx10/vendor/llama-decode-final`
- Candidate source: `/home/dvijraicha/k2-gx10/vendor/llama-decode-q6-soa`
- Candidate differs from accepted source only in
  `ggml/src/ggml-cuda/mmvq.cu`.
- Upstream `/home/dvijraicha/llama.cpp` was not modified.
- Candidate source SHA-256:
  `a4eb0c520c46a1945418adf314b5d62ea612b1233e1960881dd95436d796cb68`
- Harness SHA-256:
  `cf2e635b247503f29257a65fd386a32c5783f386eb6bdd9f96383a04b4dead0b`
- Baseline binary/CUDA library:
  `330a4acb879b18307bed6c7dedf1a20a214df16efbfa9dd9da778620232deea5` /
  `0ab764821b958fe973a6c806392690a293b91e3f30b9fd4a078f31baced29093`
- Candidate binary/CUDA library:
  `994ee83ee7232110bcfed3e8860ad7145c83bad2ffb922f708f9be13636872ba` /
  `547d3848ac6f354baf7027136f09634eddc6c0fac10245b2e187f55cc39e5fde`
- Timing runner/analyzer:
  `f75dc41cc6a81a39142b776ef0f1e84da3b5779a5987ab00911b02de569eb6a5` /
  `78056a2116d37eb840f5c8af5f37abff3ab5b4910018c14d9aeece3c34f11505`

Both builds are Release CUDA sm_121a configurations. `ldd` records in
`timing-set1/` prove that each executable resolved its own CUDA library. The
timing campaign held `/tmp/k2-gx10-gpu.lock`, found no CUDA compute peer or
loaded Ollama model, alternated B/C and C/B order, and ran unprofiled.

The frozen hypothesis said unsupported calls would retain the accepted path.
Implementation review chose the safer behavior: with `K2_Q6_SOA=1`, unsupported
shapes abort because their bytes may already be repacked. This refinement was
made before GPU execution and does not affect the two tested K2 shapes.
