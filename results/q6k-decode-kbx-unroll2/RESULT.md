# Result: rejected before GPU

The exact cc1210 Q6_K N=1 factor-two kbx unroll compiled and changed only the
intended fused non-small-K function. However, its register use increased from
48 to 72 while shared memory stayed 2,816 B. This violates the predeclared
static resource gate and risks a material occupancy loss.

The nonfused kernel and all 275 other mmvq functions remain exact
encoding/resource matches to accepted Stage A. The unroll was therefore
isolated, but not viable under the required fused resource limit.

No GPU correctness, K=6144 odd-trip test, sanitizer, timing, NCU, or full-model
run was performed. Accepted Stage A remains unchanged.
