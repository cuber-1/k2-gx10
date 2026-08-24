# Stage-A cooperative Q6_K `ql` loader result

Status: **rejected; correct and statically valid, with slower medians in both
modes and a statistically credible nonfused regression.**

## Candidate

The isolated candidate starts from accepted Stage A (eight warps plus
distance-one L2 prefetch at offsets 0/128). It changes only exact DGX Spark
Q6_K MMVQ N=1, rows-per-block 1, non-small-K to reconstruct the four `ql`
bytes cooperatively:

- phase 0: one aligned U32 per lane;
- phase 2: one aligned U32 from `ql+2+4*lane`, `shfl_up`, and a lane-0 aligned
  U16 prologue;
- lane 31's two-byte overread remains within the same Q6 block's `qh` field
  and is discarded.

Inline PTX avoids aliasing-dependent load formation. Q6 `qh`, scales, block
scale, prefetch, unpacking, and DP4A math are unchanged. Generic Q6 and N=2+
paths remain unchanged.

## Static and memory-safety gates

The corrected exact-symbol gate passed:

- fused: 48 registers, 2816 bytes shared, zero local/stack;
- nonfused: 40 registers, 1920 bytes shared, zero local/stack;
- no spills or shared-memory increase;
- fused retains 21 `LDG`, four 0/128 prefetch hints, Q8 CSE, and four DP4As;
- old paired-U16 `ql` loads are absent; `qh`, scale, and block-scale loads are
  unchanged;
- N=2 has an identical instruction-only SHA-256.

Attempt 1 branched before loading and duplicated the U32 path, producing 23
fused `LDG`; it was stopped at the static gate. The successful build and cubin
that failed this gate are preserved separately. The sole correction selected the address before one
shared U32 load and did not add another optimization. See `DEVICE_RESOURCE.md`.

The host reconstruction proof passed 107,520 exhaustive single-byte basis
cases plus 2,048 dense deterministic cases over both phases and every lane.
Bounded fused-N1 compute-sanitizer passed with zero errors and zero leaks.

## Direct GPU-output correctness

The same result-local executable was resolved explicitly against Stage-A and
candidate libraries. Every output passed byte-for-byte `cmp`, SHA-256 equality,
and the existing CPU-reference NMSE check:

| case | bytes | shared SHA-256 |
|---|---:|---|
| N=1 fused | 114688 | `75c577fb53daa5fc2e92f6f65c29edb89289930f00489ad646b82c472c3442f4` |
| N=1 nonfused | 114688 | `b40de465b21661b7b222808f9ceff44e95c6add9cf7f099cf65dd2581f9603df` |
| N=2 nonfused | 229376 | `37d99efff61a23d2a23ee6526a1f223a6c85735e6a2262320f264860b301001b` |

## Unprofiled timing

Ten Stage-A/candidate pairs per mode were interleaved with alternating first
build. Every invocation used 20 warmups and 100 timed iterations. Positive
paired percentage means the candidate is faster. Intervals are deterministic
100,000-resample percentile bootstraps of the paired median, seed
`0x4b325136`, with linear interpolation at `(n - 1) * p`.

| mode | Stage A median | Stage A IQR | candidate median | candidate IQR | paired median | paired IQR | wins | bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused N=1 | 1.6355 ms | 0.02450 ms | 1.6575 ms | 0.01175 ms | **-1.2031%** | 2.2322 pp | 4/10 | [-2.0833%, +0.3012%] |
| nonfused N=1 | 0.7865 ms | 0.01025 ms | 0.8000 ms | 0.00750 ms | **-1.4596%** | 1.7127 pp | 2/10 | [-2.8233%, -0.5043%] |

The candidate fails all fused acceptance criteria: it is below +1.0%, wins
only 4/10 pairs, and its interval includes zero. Nonfused shows a statistically
credible regression.

## Decision

Reject the cooperative loader and keep Stage A accepted. The load-count idea
does not offset its phase dispatch, shuffle, lane-0 prologue, reconstruction,
and reconvergence/dependency cost; the lower nonfused register count likewise
does not translate into speed. Per the predeclared protocol, no fresh repeat,
full-model benchmark, or Nsight Compute run was performed. The isolated
source, build, raw timing, failed first lowering, and analysis artifacts are
retained to prevent repetition.
