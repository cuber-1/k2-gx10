# DGX Spark `__vsub4` device/resource gate

Device: NVIDIA GB10, Blackwell, compute capability 12.1; build target `sm_121a`.
Driver-query facts: 48 SMs, 1536 maximum resident threads/SM, 65536 32-bit
registers/SM, and 24 maximum resident blocks/SM. CUDA toolkit: 13.0.88.

## Exact-symbol extraction

Both libraries' ELF 38 (`sm_121a`) was extracted as a standalone cubin. It is
the cubin containing the exact Q6_K MMVQ N=1 symbols. Each exact function was
then sliced from `nvdisasm --print-code`; analysis did not rely on a substring
match across the full multi-cubin library.

- baseline cubin SHA-256:
  `6590b1697d77314739a02ea0c2a30518008904f4fa69f7a61bf384c5dffb0953`
- candidate cubin SHA-256:
  `174cb5f77f9899f86de0026ad929d83aaf74e9e11dd3f53f405786595cf35ef7`
- exact nonfused symbol:
  `_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj`
- exact fused symbol:
  `_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj`

## Resource comparison

| exact specialization | baseline | candidate | decision |
|---|---|---|---|
| N=1 nonfused | 48 regs, 1920 B shared, 0 local/stack | 44 regs, 1920 B shared, 0 local/stack | pass |
| N=1 fused | 46 regs, 2816 B shared, 0 local/stack | 46 regs, 2816 B shared, 0 local/stack | pass |

## SASS gate

The fused exact function retains four `IDP.4A.S8.S8` instructions and has zero
`LDL`/`STL` instructions in both builds. Its total encoded instruction count
drops from 392 to 368. In each up/gate subtract pair, the saturating baseline's
multi-instruction clamp/permute tail is gone. The candidate lowers each proven
0..63 packed subtraction to an `IADD3` plus a `LOP3.LUT` sign correction before
the same DP4A. Thus the intended nonsaturating substitution survives integrated
compilation without a register increase or spill.

Gate decision: **PASS; correctness testing is authorized.**

Raw evidence is under `exact-cubin/`, including standalone cubins, full and
exact-function SASS, resource dumps, and baseline/candidate diffs.
