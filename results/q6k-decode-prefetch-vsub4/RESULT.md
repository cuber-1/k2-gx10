# Stage-A Q6_K prefetch plus `__vsub4` interaction result

Status: **rejected; correct and statically cheaper, but the incremental fused
gain is below the predeclared 1.0% threshold.**

## Candidate

The isolated candidate starts from accepted Stage A and changes only the
proven Q6_K MMVQ packed subtract. Exact DGX Spark CUDA device code uses
`__vsub4`; HIP, MUSA, and every other CUDA architecture retain `__vsubss4`.
The accepted eight-warp configuration and distance-one L2 hints at offsets
0/128 are byte-identical to Stage A. No other optimization is combined.

## Instruction/resource gate

The exact-symbol gate passed:

- fused remains at 48 registers, 2816 bytes shared, and zero local/stack;
- nonfused remains at 56 registers, 1920 bytes shared, and zero local/stack;
- no spills are present;
- all four fused Stage-A 0/128 `CCTL.E.PF2` sites remain;
- fused retains 21 static `LDG`, Q8 CSE, and four `IDP.4A.S8.S8`;
- the saturating subtract tail is removed.

See `DEVICE_RESOURCE.md` and `exact-cubin/`.

## Direct GPU-output correctness

The result-local comparator was built against Stage A, then the same executable
was run with explicit Stage-A and candidate library paths. Every output passed
byte-for-byte `cmp` and SHA-256 equality:

| case | bytes | shared SHA-256 |
|---|---:|---|
| N=1 fused | 114688 | `75c577fb53daa5fc2e92f6f65c29edb89289930f00489ad646b82c472c3442f4` |
| N=1 nonfused | 114688 | `b40de465b21661b7b222808f9ceff44e95c6add9cf7f099cf65dd2581f9603df` |
| N=2 nonfused | 229376 | `37d99efff61a23d2a23ee6526a1f223a6c85735e6a2262320f264860b301001b` |

The edited shared Q6 MMVQ helper also changes N=2 and MoE code generation.
N=2 is included here as correctness coverage, not as an unaffected binary
control.

All six runs also passed their CPU-reference NMSE checks.

## Unprofiled timing

Ten Stage-A/candidate pairs per mode were interleaved with alternating first
build. Every invocation used 20 warmups and 100 timed iterations. Positive
paired percentage means the interaction candidate is faster. Intervals are
deterministic 100,000-resample percentile bootstraps of the paired median,
seed `0x4b325136`, using linear interpolation at `(n - 1) * p`.

| mode | Stage A median | Stage A IQR | candidate median | candidate IQR | paired median | paired IQR | wins | bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused N=1 | 1.6485 ms | 0.00675 ms | 1.6375 ms | 0.01500 ms | +0.4831% | 0.8609 pp | 8/10 | [+0.1818%, +1.2129%] |
| nonfused N=1 | 0.7845 ms | 0.01725 ms | 0.7825 ms | 0.00350 ms | +0.5753% | 1.5571 pp | 6/10 | [-0.8392%, +1.1378%] |

The fused result reaches 8/10 wins and its interval is above zero, but its
paired median is only +0.4831%, below the mandatory +1.0% threshold. All three
criteria were required. Nonfused also misses the win-count and interval gates.

## Decision

Reject the interaction candidate. Stage A remains accepted because the
incremental source complexity does not deliver the required fused gain. Per
the predeclared protocol, no fresh repeat, full-model benchmark, or Nsight
Compute run was performed. The isolated source/build/result trees are retained
and marked rejected to prevent repetition.
