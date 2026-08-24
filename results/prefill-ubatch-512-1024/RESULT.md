# Prefill `ubatch=1024` result

Status: **rejected at correctness; no performance claim or timing sweep.**

## Gate setup

A result-local harness loaded the accepted prefetch-server build and cached
four-shard K2-Think-V2 Q6_K model exactly once. It created fresh contexts with
identical parameters except `n_ubatch=512` or 1024:

- exact same saved 2048-token vector;
- `n_batch=2048`, context 2048, Flash Attention enabled, all 99 layers on GPU;
- one final logits row per decode; 250,112 finite logits;
- no vendor or upstream source edits.

## Initial correctness result

Both decodes succeeded and both selected argmax token 42, but the numerical
difference exceeded both predeclared limits:

| metric | observed | gate | result |
|---|---:|---:|---|
| NMSE | **0.00160124428** | <=0.000001 | fail (1601x limit) |
| maximum absolute difference | 1.03129768 | — | — |
| baseline maximum absolute logit | 24.6624870 | — | — |
| normalized maximum absolute difference | **4.181645%** | <=1% | fail |
| argmax | 42 / 42 | equal | pass |
| all logits finite | true | true | pass |

Set-1 output hashes:

- ubatch 512: `b97fe64e19bdb5dbc46df44c3517e8f10462aff78ebe658f8c060e035bb79a12`;
- ubatch 1024: `eb82b0b0288022385a51987bbebeac89810a47b641d419dda8a8066f01b0bf41`.

## Harness-error diagnostic

The diagnostic reused the exact saved token bytes and, in one model load,
evaluated two fresh contexts at each setting. Independent standard-library
analysis found:

| comparison | byte equal | NMSE | max-abs % | argmax |
|---|---|---:|---:|---|
| ub512 A vs B | yes | 0 | 0 | 42 / 42 |
| ub1024 A vs B | yes | 0 | 0 | 42 / 42 |
| ub512 A vs ub1024 A | no | 0.00160124428 | 4.181645% | 42 / 42 |
| ub512 B vs ub1024 B | no | 0.00160124428 | 4.181645% | 42 / 42 |

The diagnostic hashes also exactly match set 1. Thus each setting is bitwise
deterministic across fresh contexts and separate model-load processes, while
the cross-setting delta is fixed and reproducible. This rules out ordinary
harness/run nondeterminism. Runtime logs prove distinct 512- and 1024-token
graph reservation/chunking while fused Gated Delta Net and Lightning Indexer
paths remain enabled, so the evidence supports physical-partition/reduction
order as the source of the numerical drift. It does not establish which
individual operation dominates the drift.

## Decision

Keep accepted `ubatch=512`; reject 1024 under the stated numerical contract.
The initial correctness gate is mandatory even though argmax matches. The
reviewed one-model-load `llama-bench` sweep, continuous telemetry, timing
statistics, repeat, and profiling were not run. Raw logits, exact tokens,
hashes, logs, source, binaries, process checks, and independent analysis are
preserved under `correctness/`.
