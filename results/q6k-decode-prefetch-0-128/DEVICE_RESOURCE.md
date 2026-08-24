# Stage-A prefetch device/resource gate

Device: NVIDIA GB10, Blackwell, compute capability 12.1; 48 SMs, 1536 maximum
resident threads/SM, 65536 registers/SM, and 24 maximum resident blocks/SM.
Build target: `sm_121a`; CUDA toolkit 13.0.88.

## Exact extraction

Standalone ELF 38 was extracted from both libraries because its text table
contains the exact Q6_K MMVQ N=1 fused and nonfused symbols. Exact functions
were sliced from `nvdisasm --print-code` output before instruction counting.

- baseline cubin SHA-256:
  `6590b1697d77314739a02ea0c2a30518008904f4fa69f7a61bf384c5dffb0953`
- candidate cubin SHA-256:
  `33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8`

## Resources

| exact specialization | baseline | candidate | gate |
|---|---|---|---|
| Q6_K N=1 nonfused | 48 regs, 1920 B shared, 0 local/stack | 56 regs, 1920 B shared, 0 local/stack | pass; timing required |
| Q6_K N=1 fused | 46 regs, 2816 B shared, 0 local/stack | 48 regs, 2816 B shared, 0 local/stack | pass (limit 48) |

Neither exact function contains `LDL` or `STL`; there are no spills. Fused 48
registers and baseline 46 both permit five 256-thread blocks per SM after warp
register allocation, so fused theoretical occupancy remains 83.33%. Nonfused
56 registers permits four blocks (66.67%), making the nonfused regression gate
important.

## SASS

PTX `prefetch.global.L2` lowers to `CCTL.E.PF2` on `sm_121a`.

- Fused has four static hints in the loop: an unconditional pair at effective
  next-block offsets 0/128 for `vx`, followed by a `use_gate`-predicated pair at
  the same offsets for `vgate`. The surrounding lane and next-block predicates
  ensure only lane 0 of each warp issues them for a nonterminal iteration.
- Nonfused has only `vx` hint pairs. Compiler loop versioning creates three
  static copies, but every executable nonterminal path contains exactly one
  pair; it does not dynamically execute all three pairs per iteration.
- Fused retains exactly four `IDP.4A.S8.S8` instructions.
- The two Q8 packed activation operands are loaded once and reused by both the
  up DP4As and the gate DP4As; total fused `LDG` count remains 21, matching the
  accepted baseline. Q8 CSE is retained.

Gate decision: **PASS; correctness and unprofiled timing are authorized.**

Raw standalone cubins, exact-function SASS, resource reports, and diffs are in
`exact-cubin/`.
