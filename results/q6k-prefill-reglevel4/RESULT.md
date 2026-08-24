# CUDA 13 Q6_K MMQ register-usage-level 4 result

Status: **REJECTED AT STATIC GATE**.

The source-scoped ptxas option was applied exactly as intended, but CUDA 13
generated the same Q6 device code and resources as its default register-usage
level 5. For `mul_mat_q<Q6_K,128,false>`, both builds use 255 registers, a
64-byte stack, 32 LDL, 27 STL, 512 IMMA, and 1024 bytes shared memory. J32 and
all Stream-K fixup anchors are also unchanged.

The compile-isolation gates pass:

- only Q6_K compilation receives the level-4 option;
- Q5_K and Q8_0 compile commands normalize identically to baseline;
- device-link propagation is absent;
- path/metadata-normalized embedded PTX is identical;
- all 64 Q6 functions have identical ordered 128-bit SASS encodings.

Nevertheless, the required material static improvement is absent. Lowering the
ptxas heuristic level is therefore a no-op for this CUDA 13 Q6 cubin, and the
candidate is rejected before GPU execution.

No correctness run, raw GPU output, timing, Nsight Systems, Nsight Compute, or
full-model command was run. Complete hypothesis, source diff, configure/build
logs, executed commands, hashes, raw/normalized PTX, exact cubins, resources,
SASS, and normalized encoding manifests are preserved here. Accepted source,
server source, and `/home/dvijraicha/llama.cpp` were not modified.
