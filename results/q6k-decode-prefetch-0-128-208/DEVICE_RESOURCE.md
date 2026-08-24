# Stage-B prefetch device/resource gate

Device: NVIDIA GB10, compute capability 12.1. Build target: `sm_121a`; CUDA
toolkit 13.0.88.

Standalone ELF 38 was extracted from the Stage-A and Stage-B libraries because
it contains the exact Q6_K MMVQ N=1 functions. Exact functions were sliced
from `nvdisasm --print-code` output before instruction counting.

| exact specialization | Stage A | Stage B | gate |
|---|---|---|---|
| Q6_K N=1 nonfused | 56 regs, 1920 B shared, 0 local/stack | 56 regs, 1920 B shared, 0 local/stack | pass |
| Q6_K N=1 fused | 48 regs, 2816 B shared, 0 local/stack | 48 regs, 2816 B shared, 0 local/stack | pass |

Neither exact Stage-B function contains `LDL` or `STL`; there are no spills.

Stage B lowers the new hint to `CCTL.E.PF2`. Fused contains exactly six static
sites: three for `vx` and three runtime-gate-predicated sites for `vgate`.
Their effective addresses differ by 0, 128, and 208 bytes. Nonfused compiler
loop versioning produces three copies of a three-site sequence, but any
executable nonterminal path issues exactly three sites at offsets 0, 128, and
208. This is the same versioning behavior seen in Stage A with two-site
sequences.

Fused retains 21 static `LDG` instructions, four `IDP.4A.S8.S8` instructions,
and reuses the same Q8 packed activation registers in both the up and gate dot
paths. Q8 CSE is retained.

Exact cubin hashes:

- Stage A: `33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8`;
- Stage B: `6ad842b108eb45986de14e4bcd64d2663ae8219c9ce0ddaac04b6c913c905476`.

Gate decision: **pass; direct-output correctness and unprofiled timing were
authorized.** Raw cubins, SASS, resource reports, counts, and diffs are in
`exact-cubin/`.
