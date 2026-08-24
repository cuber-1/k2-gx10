# Cooperative Q6_K `ql` loader device/resource gate

Device: NVIDIA GB10, compute capability 12.1. Build target: `sm_121a`; CUDA
toolkit 13.0.88.

Standalone ELF 38 was extracted from Stage A and the candidate because it
contains the exact Q6_K MMVQ symbols. Exact functions were sliced from
`nvdisasm --print-code` output before counting.

## Corrected static gate

| exact specialization | Stage A | candidate | gate |
|---|---|---|---|
| Q6_K N=1 nonfused | 56 regs, 1920 B shared, 0 local/stack | 40 regs, 1920 B shared, 0 local/stack | pass |
| Q6_K N=1 fused | 48 regs, 2816 B shared, 0 local/stack | 48 regs, 2816 B shared, 0 local/stack | pass |
| Q6_K N=2 nonfused control | 56 regs, 2560 B shared, 0 local/stack | 56 regs, 2560 B shared, 0 local/stack | unchanged |

No selected candidate symbol contains `LDL` or `STL`; there are no spills or
shared-memory increases.

The corrected exact fused symbol retains:

- four Stage-A `CCTL.E.PF2` sites at effective offsets 0 and 128 for `vx` and
  runtime-enabled `vgate`;
- 21 static `LDG`, matching Stage A;
- four `IDP.4A.S8.S8` operations;
- the same packed Q8 activation registers reused by the up and gate dots.

The `ql` path now contains one aligned U32 load per matrix invocation. The
uniform phase bit selects `ql + phase + 4*lane` before that load. Phase 0 uses
the word directly. Phase 2 executes one `SHFL.UP`, with a lane-0-only aligned
U16 prologue; lane 31's U32 spans `ql[126:127]` and `qh[0:1]`, but only the
first two bytes contribute. The old paired U16 loads at `ql+4*lane` and
`ql+4*lane+2` are absent. The remaining paired U16 loads at block offsets
0x80/0x82 are the unchanged `qh` loads. Scale loads at 0xc0/0xc4 and block
scale loads at 0xd0 are also unchanged.

The N=2 control has the same 22 `LDG`, 14 U16 loads, 20 shuffles, and eight
DP4As. Its instruction-only SHA-256 is identical for Stage A and candidate:
`672c951d7145cabc247974a0361bfab3cf5918161d6bafb4e6fcbc5b798a40b1`.
The only textual disassembly differences are the compiler's global label
numbers.

Exact cubin hashes:

- Stage A: `33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8`;
- corrected candidate:
  `5822e7ae8d6b59e955af7076d2d416d4f5661201959c4cca1001492b2e5a5459`.

## Preserved failed lowering

The first source lowering branched before the aligned U32 load, so the compiler
duplicated that load across phase 0 and phase 2 for both `vx` and `vgate`. It
used 23 fused `LDG` and failed the predeclared 21-load gate despite passing
register/shared-memory limits. Its build and exact cubin/SASS are preserved as
`build-attempt1.log` and `exact-cubin-attempt1/`. The only correction was to
select the aligned address before one shared U32 load; no second optimization
was added.

Corrected gate decision: **pass; sanitizer, direct-output correctness, and
unprofiled timing are authorized.**
