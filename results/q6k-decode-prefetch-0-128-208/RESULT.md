# Q6_K L2 prefetch Stage B result

Status: **rejected; the additional +208 hint is neutral versus Stage A.**

## Candidate

Stage B adds only one `prefetch.global.L2` hint at byte offset 208 to the
accepted Stage-A helper. Existing 0/128 hints, prefetch distance, loop guards,
warp/block mapping, and arithmetic are unchanged. Both `vx` and
runtime-enabled `vgate` inherit the third hint. No phase condition, distance
change, `__vsub4`, or other candidate is combined.

## Instruction/resource gate

The exact-symbol gate passed:

- fused: 48 registers, 2816 bytes shared, no local/stack or spills;
- nonfused: 56 registers, 1920 bytes shared, no local/stack or spills;
- fused has six 0/128/208 `CCTL.E.PF2` sites;
- each executable nonfused loop-version path has three 0/128/208 sites;
- fused retains Q8 CSE, 21 static `LDG`, and four `IDP.4A.S8.S8`.

See `DEVICE_RESOURCE.md` and `exact-cubin/`.

## Direct GPU-output correctness

The result-local output comparator was built against Stage A, then the same
executable was run with explicit Stage-A and Stage-B library paths. `ldd` and
SHA-256 provenance are preserved. Every comparison passed byte for byte:

| case | bytes | shared SHA-256 |
|---|---:|---|
| N=1 fused | 114688 | `75c577fb53daa5fc2e92f6f65c29edb89289930f00489ad646b82c472c3442f4` |
| N=1 nonfused | 114688 | `b40de465b21661b7b222808f9ceff44e95c6add9cf7f099cf65dd2581f9603df` |
| unchanged N=2 nonfused | 229376 | `37d99efff61a23d2a23ee6526a1f223a6c85735e6a2262320f264860b301001b` |

All six runs also passed their CPU-reference NMSE checks.

## Unprofiled timing

Ten Stage-A/Stage-B pairs per mode were interleaved with alternating first
build. Every invocation used 20 warmups and 100 timed iterations. Positive
paired percentage means Stage B faster. Intervals are deterministic 100,000
resample percentile bootstraps of the paired median, seed `0x4b325136`, using
linear interpolation at `(n - 1) * p`.

| mode | Stage A median | Stage A IQR | Stage B median | Stage B IQR | paired median | paired IQR | wins | bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused N=1 | 1.6565 ms | 0.01575 ms | 1.6535 ms | 0.01700 ms | +0.0612% | 1.5973 pp | 5/10 | [-0.8235%, +1.3176%] |
| nonfused N=1 | 0.7850 ms | 0.00400 ms | 0.7805 ms | 0.00425 ms | +0.5091% | 1.2649 pp | 6/10 | [-0.1287%, +1.4018%] |

The fused acceptance requirements were at least +1.0% paired median, at least
8/10 wins, and an interval strictly above zero. Stage B misses all three.

## Decision

Reject Stage B as neutral. Per the predeclared rule, Stage A is simpler and
remains accepted. No fresh repeat, full-model benchmark, or Nsight Compute run
is warranted. The isolated Stage-B source/build/result trees are retained and
clearly marked rejected so this idea is not repeated.

Independent review recomputed the paired statistics and confirmed the
rejection. Timing executable hashes and resolved library paths are preserved;
GPU-exclusivity checks were performed but their command output and separate
timing-time CUDA-library hashes were not captured. These provenance gaps do
not weaken a neutral/reject decision, but later candidates should preserve
those outputs explicitly.
