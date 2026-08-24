# Stage-A prefetch plus `__vsub4` device/resource gate

Device: NVIDIA GB10, compute capability 12.1. Build target: `sm_121a`; CUDA
toolkit 13.0.88.

Standalone ELF 38 was extracted from both libraries because it contains the
exact Q6_K MMVQ N=1 fused and nonfused symbols. Exact functions were sliced
from `nvdisasm --print-code` output before counting.

| exact specialization | Stage A | interaction candidate | gate |
|---|---|---|---|
| Q6_K N=1 nonfused | 56 regs, 1920 B shared, 0 local/stack | 56 regs, 1920 B shared, 0 local/stack | pass |
| Q6_K N=1 fused | 48 regs, 2816 B shared, 0 local/stack | 48 regs, 2816 B shared, 0 local/stack | pass |

Neither candidate symbol contains `LDL` or `STL`; there are no spills.

The exact fused candidate retains all four Stage-A `CCTL.E.PF2` sites at
effective 0/128 offsets, 21 static `LDG`, and four `IDP.4A.S8.S8` operations.
The two packed Q8 activation operands remain loaded once and reused by the up
and gate dot paths. Q8 CSE is retained.

The exact fused function shrinks from 432 to 416 disassembly lines, and the
nonfused function from 373 to 341. The long saturating subtract tail is
replaced by the non-saturating packed subtract lowering (`IADD3` plus
`LOP3.LUT` sign correction); its `PRMT`/clamp sequence is absent. This matches
the previously proven standalone lowering while preserving the Stage-A
prefetch instructions.

Exact cubin hashes:

- Stage A: `33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8`;
- interaction candidate:
  `e469a1a0685bf57fe92b097138f7bd555fb052c60f0d359969f1b80de45ce26c`.

Gate decision: **pass; direct-output correctness and unprofiled timing were
authorized.** Raw cubins, exact-function SASS, resource reports, and diffs are
in `exact-cubin/`.
