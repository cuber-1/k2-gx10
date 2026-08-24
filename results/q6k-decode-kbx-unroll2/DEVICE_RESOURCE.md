# Device/resource evidence

Status: rejected before GPU.

The exact sm_121a candidate fused kernel uses 72 registers, zero stack/local
spills, and 2,816 B shared memory. Accepted Stage A uses 48 registers with the
same stack/shared values. The candidate therefore violates the mandatory
fused <=48-register gate by 24 registers.

The nonfused symbol remains exactly Stage A: 56 registers, zero stack/local
spills, and 1,920 B shared. Whole-cubin comparison proves all 275 other
functions are exact encoding/resource matches; only fused Q6_K N1 non-small-K
changed.

Raw resource reports, exact cubins/PTX, hex SASS, extracted fused functions,
and the 276-function mapping are under `static/`.
