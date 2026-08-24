# Compile and static resource record

Status: rejected at the mandatory compile gate; no GPU kernel was run.

- Fresh candidate configuration: Release, requested CUDA architecture 121,
  resolved by the project to exact `sm_121a`; isolated source root
  `vendor/llama-prefill-q8-index-cse`.
- Baseline is the frozen accepted Q6 object/cubin. Baseline cubin SHA256:
  `a96e9e17afd5f20ee92a42cb5d8d2e3c708792ba5412967736d037f6d395e37e`.
- Candidate Q6 object SHA256:
  `bb9194df104daf9b141b0de5566283e6750dba584ad6f7b16cc7eb5c6a95d321`;
  extracted candidate cubin SHA256:
  `fcb3a8b2410364059d18d2834a6249c236848c5474d86cdb9489dccf778a423a`.

## J=128, non-fallback main symbol

| metric | baseline | candidate | gate |
|---|---:|---:|---:|
| instructions | 8,744 | 8,736 | recurrence must also reduce spills |
| registers | 255 | 255 | no worsening |
| stack | 64 B | 80 B | <=64 B |
| LDL | 32 | 34 | <=30 |
| STL | 27 | 29 | <=26 |
| IMMA | 512 | 512 | unchanged |
| FFMA | 1,280 | 1,280 | unchanged |
| barriers | 13 | 13 | unchanged |
| branch instructions | 80 | 80 | unchanged topology |
| LDG | 215 | 215 | unchanged |
| LDS + LDSM | 488 | 488 | unchanged |
| STS | 145 | 145 | unchanged |

The candidate shortens index code by eight instructions but allocates 16 more
stack bytes and adds two local loads plus two local stores. The terminal region
also exposes five late local sum reloads where the accepted sequence has four,
the opposite of the required <=3. This is an immediate hard failure regardless
of the algebraic CSE.

## Isolation audit

Ordered 128-bit instruction streams show only J128,false real kernel code
changes. Forty-seven of 64 functions are encoding-identical, including all 32
Stream-K fixup kernels, J128,true, J32,false, J32,true, and all other real
non-target specializations. The remaining 16 non-target entries are the
80-word `NO_DEVICE_CODE` placeholders: each has exactly one differing encoded
word, the `__LINE__` diagnostic immediate shifting from accepted line 959 to
candidate line 969 because ten source lines were inserted before
`NO_DEVICE_CODE`. `static/non-target-line-normalization.tsv` records this
one-word diagnostic difference for every placeholder. After that known line
normalization, all 63 non-target functions are semantically identical.
`static/normalized-function-encoding-compare.tsv` contains the regenerated
masked hashes and reports exactly 63 normalized-identical functions and one
changed target.

Neighbor non-Q6 translation units were not compiled because the mandatory Q6
spill gate had already hard-failed. Their isolation is supported by the exact
constexpr type guard and verbatim false arms, not by a candidate cubin audit;
this limitation is explicit rather than presenting source proof as binary
identity.
