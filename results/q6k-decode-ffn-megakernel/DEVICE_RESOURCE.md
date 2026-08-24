# Device resource gate

The exact added sm_121a symbol is:

`k2_ffn_megakernel_q6_K(const block_q6_K *, const block_q6_K *, const block_q6_K *, const float *, block_q8_1 *, float *, block_q8_1 *, float *)`

| Build | REG | STACK | SHARED | LOCAL | Result |
|---|---:|---:|---:|---:|---|
| Initial | 64 | 0 | 2816 B | 0 | Static pass; timed and rejected |
| Launch-bounds 5 | 48 | 0 | 2816 B | 0 | Static pass; timed and rejected |

The initial Nsight Systems identity capture reports grid `(192,1,1)`, block `(32,8,1)`, 64 registers/thread, and zero
local memory. With 48 SMs this is four CTAs/SM. The follow-up's 48-register allocation permits the requested five
256-thread CTAs/SM (240 cooperative CTAs total) without stack/local traffic.

An ordered 128-bit encoding audit compared each pre-existing function in the accepted final sm_121a cubin to both
candidate cubins. All 276/276 existing functions are exact matches in each candidate. The sole extra symbol is the
megakernel above.
