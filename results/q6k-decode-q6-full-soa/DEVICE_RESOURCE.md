# Full-field Q6 SoA device resources

| Kernel | Registers | Stack/local | Shared | Instructions | LDG | CCTL.PF2 | IDP.4A |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=8192 fused | 48 | 0/0 | 2,816 B | 264 | 18 | 4 | 4 |
| K=8192 nonfused | 40 | 0/0 | 1,920 B | 176 | 11 | 2 | 2 |
| K=28672 nonfused | 40 | 0/0 | 1,920 B | 176 | 11 | 2 | 2 |

The two prefetch hints per active matrix target the next block's aligned `ql`
and `qh` starts. Existing accepted row-major Q6 N=1 instruction streams retain
hashes `a3e9c2b8...` fused and `57faa2ce...` nonfused and compare exactly.
