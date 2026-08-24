# Cooperative Q6_K `qh` loader device/resource gate

Device: NVIDIA GB10, compute capability 12.1. Build target: `sm_121a`; CUDA
toolkit 13.0.88.

Standalone ELF 38 was extracted from Stage A and the candidate because it
contains the exact Q6_K MMVQ specializations. Functions were sliced from
`nvdisasm --print-code` output before counting.

## Resources

| exact specialization | Stage A | candidate | gate |
|---|---|---|---|
| Q6_K N=1 nonfused | 56 regs, 1920 B shared, 0 local/stack | 40 regs, 1920 B shared, 0 local/stack | pass |
| Q6_K N=1 fused | 48 regs, 2816 B shared, 0 local/stack | 48 regs, 2816 B shared, 0 local/stack | pass |
| Q6_K N=2 nonfused control | 56 regs, 2560 B shared, 0 local/stack | 56 regs, 2560 B shared, 0 local/stack | unchanged |

No selected candidate symbol contains `LDL` or `STL`; there are no spills or
shared-memory increases.

## Exact fused instruction audit

The candidate reduces fused static global loads from 21 to 19 and U16 loads
from 12 to 8. In each up/gate dot path, the old two U16 `qh` loads at block
offsets 0x80/0x82 are absent and exactly one aligned U32 load is present:

- first matrix U32 at SASS `/*0920*/`, followed by XOR-8 shuffle at `/*09b0*/`;
- second matrix U32 at `/*0dc0*/`, followed by XOR-8 shuffle at `/*0e30*/`.

The remaining paired U16 loads at `/*08f0*/`/`/*0910*/` and
`/*0d90*/`/`/*0db0*/` are the unchanged `ql` loads. Scale loads remain at
effective block offsets 0xc0/0xc4 and `d` at 0xd0. All four Stage-A
`CCTL.E.PF2` hints remain. Q8 activation values are still commoned between up
and gate, and the loop retains four `IDP.4A.S8.S8` operations.

The phase-2 path contains one XOR-8 exchange per matrix. The compiler emits
warp-convergence bookkeeping for the uniform phase branch; this is a possible
runtime cost to be decided only by unprofiled timing.

## N=2 control

N=2 retains 22 `LDG`, 14 U16 loads, 20 shuffles, and eight DP4As. Its
instruction-only SHA-256 is identical for Stage A and candidate:
`672c951d7145cabc247974a0361bfab3cf5918161d6bafb4e6fcbc5b798a40b1`.
Textual disassembly differences are limited to compiler global label numbers.

Exact cubin hashes:

- Stage A: `33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8`;
- candidate: `3bf4acf77c8b16e65a62202b1bca00e7799057e9ace32119d8449566d9a8512d`.

Static gate decision: **pass; sanitizer, direct-output correctness, and
unprofiled timing are authorized.**
