# Q6_K J128 factor-four partial unroll result

Status: **REJECTED AT STATIC GATE**.

## Conclusion

The pragma did produce a real partially unrolled target loop, but it made the
already resource-saturated production specialization substantially worse:
`mul_mat_q<Q6_K,128,false>` changed from 255 registers and a 64-byte stack to
254 registers and a 256-byte stack. Static local loads rose from 32 to 208 and
local stores from 27 to 160. This fails the predeclared no-worse-spill and
material-local-traffic-reduction gate, so no correctness or performance run is
permitted.

The candidate passes the non-target isolation gate. Exact-cubin extraction
found 32 Q6 `mul_mat_q` instantiations; normalized hashes of each function's
ordered 128-bit instruction words show all 31 non-target streams are identical
and only J128/nonfallback changes. Raw `nvdisasm` text hashes differ because
shortening the target changes file-wide comment-column padding, so they are
retained only as formatting-sensitive provenance.

## Evidence

- Hypothesis and numeric gates were written before source/harness edits in
  `HYPOTHESIS.md`.
- Both baseline and candidate were clean, fresh builds against the same frozen
  harness; accepted builds were not rebuilt.
- CMake resolved both to `sm_121a`; complete configure/build logs and binary,
  library, CMake-cache, source, harness, patch, cubin, and SASS hashes are
  preserved in this result directory.
- The frozen candidate diff contains only the cc1210 pragma guard immediately
  before the outer `j0` loop in
  `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma`.
- A first wrong-anchor draft on the DP4A `k01` loop was fully reverted before
  configure/build and is documented in `WRONG-ANCHOR-ATTEMPT.md`.
- Primary harness dry-run reports grid 48, Q8 workspace 4,737,024 bytes,
  Stream-K fixup 3,145,728 bytes, and guarded total 561,039,360 bytes.

## Work intentionally not run

No CUDA graph, correctness comparison, raw GPU output, performance timing,
Nsight Systems, Nsight Compute, or full-model command was run. The static gate
is independently decisive. The isolated candidate tree/build remain available
only as rejected evidence; accepted source, server source, and
`/home/dvijraicha/llama.cpp` were not modified.
