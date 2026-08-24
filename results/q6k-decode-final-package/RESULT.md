# Final DGX Spark Q6_K decode package

## Result

ACCEPT for packaging. The consolidated patch is
`patches/q6k-gb10-decode-final.patch` (SHA-256
`8944a9c7ea9798ddf843d4878a0a9e574b00e51bbc49bf883d57705596bb8d31`).
It contains the two measured decode changes—Q6_K N=1 eight warps and
distance-one L2 prefetches at byte offsets 0 and 128—and narrows both special
table selectors to exactly NVIDIA DGX Spark cc 1210.

The final sm_121a MMVQ cubin is byte-identical to the accepted measured Stage-A
cubin (both SHA-256
`33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8`).
All 276 logical kernel symbols, their resources, and their ordered instruction
encodings compare byte for byte. Therefore the selector scope correction did
not change any code executed on the measured GB10 target, and no GPU or
full-model rerun was needed.

The accepted measured result remains:

- Full model: 3.549480 to 3.920885 tok/s median, +10.4636%, with disjoint
  10-sample ranges (3.537980–3.556080 versus 3.906080–3.929290 tok/s).
- Fused microbenchmark repeats: +14.49% and +14.44%, 10/10 paired wins in each.
- Nonfused microbenchmark repeats: +12.13% in each, 10/10 paired wins in each.
- Correctness: byte-identical outputs for fused N=1, nonfused N=1, and N=2.

Timing and correctness evidence remains in
`results/q6k-decode-prefetch-0-128/`. This directory records only final
packaging/static-equivalence evidence and does not claim a new timing result.

## Scope

The isolated final source is
`vendor/llama-decode-final/ggml/src/ggml-cuda/mmvq.cu`. Relative to the
accepted Stage-A checkout, this is the only changed file. The host selector is
`cc == GGML_CUDA_CC_DGX_SPARK`; the device selector requires CUDA (not HIP or
MUSA) and exact `__CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK`. Q6_K N=1 remains
eight warps, all rows-table entries remain unchanged, and the accepted
prefetch offsets remain 0/128.

The original `/home/dvijraicha/llama.cpp` source was not modified.

## Non-1210 control

An actual sm_120a compilation of the final source selected the generic path:
the Q6_K N=1 nonfused/fused kernels use 48/46 registers and 1408/1792 bytes of
shared memory, versus the cc1210 eight-warp specializations at 56/48 registers
and 1920/2816 bytes. The complete sm_120a MMVQ SASS contains zero
`CCTL.E.PF2` hints. Host AArch64 disassembly independently contains an exact
`cmp w24, #0x4ba` (1210) before entering the special-table branch.

## Apply

```bash
cd /home/dvijraicha/llama.cpp
git apply --check /home/dvijraicha/k2-gx10/patches/q6k-gb10-decode-final.patch
git apply /home/dvijraicha/k2-gx10/patches/q6k-gb10-decode-final.patch
cmake --build build --target llama-server llama-bench --parallel 4
```

