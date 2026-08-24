# Device and static resource record

Status: static-gate rejection; no GPU kernel was executed.

- Device reported by `nvidia-smi`: NVIDIA GB10, compute capability 12.1,
  driver 580.142, PCI 0000000F:01:00.0.
- CUDA compiler: NVIDIA CUDA 13.0.88.
- Both fresh configurations requested architecture 121 and CMake resolved it
  to `sm_121a`/`121a-real`.
- Baseline tree: `vendor/llama-decode-prefetch-server`.
- Candidate tree: `vendor/llama-prefill-unroll4`.
- Frozen root harness SHA256:
  `1378536c577aac6d8d31d0cd63e2b676480052982dac4d6274c0e32923627bbb`.
- Frozen baseline `mmq-vec-dot.cuh` SHA256:
  `e118f96fc959be2938e61a574561fff57b8e0c42a71d22ae8e97139a4fba34f5`.
- Frozen candidate `mmq-vec-dot.cuh` SHA256:
  `a84a8906b0494d3b0c46d8b0f9a741acbfc338c1011e9849e4df26ba0ce54c02`.

Exact `sm_121a` Q6_K cubin was ELF 120 in both fresh libraries. For
`mul_mat_q<Q6_K,128,false>`:

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Registers | 255 | 254 | -1 |
| Stack frame | 64 B | 256 B | +192 B / 4x |
| Shared memory | 1024 B | 1024 B | unchanged |
| Static instructions | 8744 | 6400 | -26.8% |
| Static LDL | 32 | 208 | +550% |
| Static STL | 27 | 160 | +492.6% |
| Static IMMA | 512 | 256 | -50% |

The IMMA reduction is static only: SASS shows a real factor-four lowering with
runtime loop backedges and two trips, so the candidate has not removed dynamic
math. The two Stream-K control paths must be normalized by their runtime trip
count. Conversely, the much larger stack and local load/store bodies are real
spill pressure and violate the predeclared gate.

Resource use and normalized instruction encodings of the other 31 Q6
`mul_mat_q` instantiations are unchanged. Across all 64 functions in the Q6
cubin, 63 non-target ordered instruction-word streams match exactly; only
J128/nonfallback changes from 8,744 to 6,400 instructions. The raw textual SASS
hashes in `static/q6-specialization-compare.tsv` are formatting-sensitive:
target shortening changes `nvdisasm`'s file-wide right-comment padding even on
identical instructions. The isolation gate therefore passes. The target spill
gate remains independently decisive.
