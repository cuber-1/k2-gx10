# Q6 field-SoA device resources

Exact sm_121a resource rows from the isolated candidate `mmvq.cu` cubin:

| Kernel | Registers | Stack | Local | Shared | Static instructions | LDG | CCTL.PF2 | IDP.4A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K=8192 fused | 48 | 0 | 0 | 2,816 B | 264 | 18 | 4 | 4 |
| K=8192 nonfused | 40 | 0 | 0 | 1,920 B | 168 | 11 | 2 | 2 |
| K=28672 nonfused | 40 | 0 | 0 | 1,920 B | 168 | 11 | 2 | 2 |

The existing accepted Q6_K N=1 fused and nonfused ordered instruction streams
match the final-package baseline exactly. Their normalized encoding hashes are
`a3e9c2b85ea37162dd347782e115350361846344fb59abe7a07b072e504a349b`
and `57faa2ce3d27e4fdd94f2836e584a1b0adafdb5ca9d871e637a14c0e05cdc01e`.

The new functions use the same eight-warp reduction geometry and existing
Q6/Q8 DP4A implementation. Prefetch sites target the next logical block's
`ql` and tail starts only when that next block exists.
