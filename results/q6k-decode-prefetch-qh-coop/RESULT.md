# Stage-A cooperative Q6_K `qh` loader result

Status: **rejected; correct and statically cheaper, but the fused gain is below
the predeclared threshold and the nonfused median is slower.**

## Candidate

The candidate starts from accepted Stage A, not the rejected cooperative `ql`
tree. Exact DGX Spark Q6_K MMVQ N=1, rows-per-block 1, non-small-K replaces two
U16 `qh` loads with one aligned U32 per lane-pair path. Phase 0 loads directly;
phase 2 has paired lanes load the aligned words around the desired bytes,
exchange with XOR-8 shuffle, and reconstruct the word. All accesses are formed
from serialized block bytes and remain within the same 210-byte Q6 object.

`ql`, scales, block scale, Q8 loads/math, Stage-A 0/128 prefetch, generic Q6,
and N=2+ are unchanged.

## Static and correctness gates

Exact fused SASS passes the gate:

- 48 registers, 2816 bytes shared, no local/stack or spills;
- static `LDG` falls 21→19 and U16 loads 12→8;
- exactly one U32 `qh` load plus one XOR-8 shuffle per up/gate path;
- old U16 `qh` pairs are absent; `ql`, scales, and `d` loads are unchanged;
- four prefetch hints, Q8 CSE, and four DP4As remain.

Nonfused uses 40 registers and 1920 bytes shared. N=2 has an identical
instruction-only SHA-256 to Stage A. See `DEVICE_RESOURCE.md`.

The host proof passed 107,520 exhaustive single-byte basis cases plus 2,048
dense deterministic warp cases over both alignment phases and all lanes. Bounded
fused-N1 compute-sanitizer passed with zero errors and zero leaks.

The same result-local executable, explicitly resolved to Stage-A and candidate
libraries, produced byte-identical GPU output and matching SHA-256 for all
required cases:

| case | bytes | shared SHA-256 |
|---|---:|---|
| N=1 fused | 114688 | `75c577fb53daa5fc2e92f6f65c29edb89289930f00489ad646b82c472c3442f4` |
| N=1 nonfused | 114688 | `b40de465b21661b7b222808f9ceff44e95c6add9cf7f099cf65dd2581f9603df` |
| N=2 nonfused | 229376 | `37d99efff61a23d2a23ee6526a1f223a6c85735e6a2262320f264860b301001b` |

## Unprofiled timing

Ten Stage-A/candidate pairs per mode used alternating first build, 20 warmups,
and 100 timed iterations. Positive paired percentage means the candidate is
faster. Intervals are deterministic 100,000-resample percentile bootstraps of
the paired median, seed `0x4b325136`, with linear `(n - 1) * p` interpolation.

| mode | Stage A median | Stage A IQR | candidate median | candidate IQR | paired median | paired IQR | wins | bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused N=1 | 1.6495 ms | 0.00650 ms | 1.6365 ms | 0.02275 ms | **+0.4543%** | 1.6498 pp | 7/10 | [-0.1824%, +1.7565%] |
| nonfused N=1 | 0.7855 ms | 0.01025 ms | 0.7950 ms | 0.00875 ms | **-1.1531%** | 1.7456 pp | 3/10 | [-1.8441%, +0.5051%] |

Fused misses all mandatory criteria: less than +1.0%, fewer than 8/10 wins,
and a confidence interval crossing zero. Nonfused trends slower.

## Decision

Reject and retain accepted Stage A. The two fewer static global loads did not
demonstrate an acceptable benefit after adding phase dispatch, XOR-8 shuffle,
reconstruction, and compiler-emitted warp-convergence bookkeeping. Per
protocol, no fresh repeat, full-model run,
or Nsight Compute collection was performed. The isolated source, build, raw
timings, SASS, correctness outputs, and analysis are retained.
