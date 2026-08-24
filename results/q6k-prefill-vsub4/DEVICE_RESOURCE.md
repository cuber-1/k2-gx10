# Build and static resource record

Status: static and correctness gates passed; primary timing rejected the
candidate with a reproducible regression.

- Device: NVIDIA GB10, compute capability 12.1, driver 580.142.
- Compiler: CUDA 13.0.88; both builds target exact `sm_121a`.
- Frozen shared harness SHA256:
  `1378536c577aac6d8d31d0cd63e2b676480052982dac4d6274c0e32923627bbb`.
- Baseline CUDA library SHA256:
  `87bece1ef046ea5a5efd14ea1865aeb18ae2086135e6bb6bc38bac9137ca395a`.
- Candidate CUDA library SHA256:
  `ce5835a62828a83291a5327e206388e4facd037e4b0ac624248e60da30a25e6c`.
- Loaded Q6 cubin is ELF 120 in both libraries. It byte-matches the exact
  separately compiled Q6 object cubin in each build: baseline
  `a96e9e17afd5f20ee92a42cb5d8d2e3c708792ba5412967736d037f6d395e37e`,
  candidate
  `c4c83464c2d9540bfc596788a7c4366e1c5cdae4bd4bf896e2ae0cb55e65247f`.

## J=128, non-fallback production specialization

| metric | baseline | candidate |
|---|---:|---:|
| instructions | 8,744 | 8,432 |
| PRMT total | 160 | 32 |
| PRMT control `0xba98` | 128 | 0 |
| PRMT control `0x8880` | 32 | 32 |
| IMMA | 512 | 512 |
| FFMA | 1,280 | 1,280 |
| LDG | 215 | 215 |
| LDS + LDSM | 488 | 488 |
| STS | 145 | 145 |
| barriers | 13 | 13 |
| LDL / STL | 32 / 27 | 32 / 27 |
| registers | 255 | 255 |
| stack / shared | 64 B / 1024 B | 64 B / 1024 B |

The candidate therefore removes the saturation-tail sequence materially while
preserving matrix math, global/shared topology, local traffic, and resources.
This is not a compiler no-op.

J=32 also removes the saturation tail and remains at 254 registers, zero stack,
and zero local loads/stores. All 32 Stream-K fixup functions have identical
ordered instruction encodings. Across the 32 main Q6 functions, the 16 real
MMA implementations change as expected and the 16 placeholder/unsupported
streams remain identical. Neighbor Q5_K, Q8_0, and Q4_K cubins have identical
normalized instruction encodings; their raw container hashes differ only due
to isolated source-root metadata.

One non-primary nuance is preserved: `J=8,false` increases from 190 to 192
registers, though it remains stack-free. This does not violate the declared
J128/J32 gates, but it is why this record does not claim every Q6 resource is
unchanged.
