# Device resource and encoding gate

The extracted final sm_121a cubin is byte-identical to the accepted measured
Stage-A cubin. Consequently all 276 logical kernels have identical ordered
instruction encodings and resources, not only the two targeted Q6_K symbols.
The complete evidence is preserved under `static/stage-a/` and `static/final/`:

| Evidence | Stage-A versus final | Final size |
|---|---|---:|
| Extracted sm_121a cubin | byte-identical, SHA-256 `33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8` | one cubin |
| Logical symbol map | byte-identical | 276 entry symbols |
| Resource map | byte-identical | 556 lines / 56,666 bytes |
| Ordered raw encodings | byte-identical | 176,806 lines / 11,759,301 bytes |

Targeted Q6_K N=1 resources:

| Target | Registers | Static shared | Stack | Local | Prefetch sites | DP4A sites |
|---|---:|---:|---:|---:|---:|---:|
| cc1210 nonfused | 56 | 1920 B | 0 | 0 | 6 static | 6 static |
| cc1210 fused | 48 | 2816 B | 0 | 0 | 4 | 4 |
| sm120a control nonfused | 48 | 1408 B | 0 | 0 | 0 | generic path |
| sm120a control fused | 46 | 1792 B | 0 | 0 | 0 | generic path |

The nonfused cc1210 count includes compiler loop versions; dynamically, lane 0
issues the accepted two distance-one hints (offsets 0 and 128) for each valid
next Q6_K block. The fused specialization has those two sites for the main
weights and two conditional sites for the gate weights. Neither target kernel
has stack or local-memory traffic.

No occupancy/utilization estimate is used as a performance claim. The accepted
normal-timing and correctness result is inherited from
`results/q6k-decode-prefetch-0-128/` and summarized in `RESULT.md`.

