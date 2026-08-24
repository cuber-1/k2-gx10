# Q6_K Blackwell N=1 rows-per-block=2 result

Decision: **REJECTED at the compile/resource gate; not benchmarked.**

Candidate source: `/home/dvijraicha/k2-gx10/vendor/llama-decode-rpb2`

Build: `/home/dvijraicha/k2-gx10/build-decode-rpb2`

Accepted baseline: `/home/dvijraicha/k2-gx10/build-decode-nw8`

The single source variable was the Blackwell Q6_K N=1 rows-per-CUDA-block
selector: 1 to 2, retaining the accepted 8-warps-per-block configuration.

| specialization | baseline regs | candidate regs | baseline shared | candidate shared | local/stack |
|---|---:|---:|---:|---:|---:|
| Q6_K N=1 nonfused | 48 | 48 | 1920 B | 2816 B | 0 / 0 |
| Q6_K N=1 fused | 46 | 56 | 2816 B | 4608 B | 0 / 0 |
| Q6_K N=2 nonfused control | 56 | 56 | 2560 B | 2560 B | 0 / 0 |

For the dominant fused 256-thread kernel, 56 registers/thread permits only four
resident blocks in the GB10's 65536-register SM file. The accepted 46-register
kernel permits five. The resulting theoretical occupancy step is 83.33% to
66.67%, which triggers the experiment's mandatory early rejection rule.

SASS additionally shows the two independent Q6_K row load/dot streams and no
`LDL`/`STL` spill traffic. Because the resource gate failed, correctness and
interleaved timings were deliberately not run; reporting a performance delta
would violate the experiment plan.

Evidence files:

- `configure.log`, `build.log`
- `device-limits.txt`
- `resource-usage-baseline-q6k-n1.txt`
- `resource-usage-q6k-n1.txt`
- `sass-q6k-n1-baseline.txt`
- `sass-q6k-n1-rpb2.txt`
