# Code walkthrough

This repository delivers a narrow llama.cpp patch plus the harnesses and evidence used to accept
it. The shortest review path is result, patch, isolated validation, then full-model validation.

## 1. Start with the accepted result

[`results/q6k-decode-combined-20260824/RESULT.md`](../results/q6k-decode-combined-20260824/RESULT.md)
compares an untouched four-warp build directly with the final eight-warp plus L2-prefetch build.
The pooled medians were 3.238285 and 3.804630 tokens/s, a 17.4890% throughput improvement and
14.8857% less steady decode time per token. All five balanced fresh-process pairs won.

The earlier staged experiments remain useful for attribution:

- four to eight warps: +4.67% full-model throughput;
- prefetch over the eight-warp baseline: +10.4636% full-model throughput; and
- the same prefetch in the isolated fused benchmark: +14.49% and +14.44% in two repeats.

## 2. Read the production patch

The complete accepted change is
[`patches/q6k-gb10-decode-final.patch`](../patches/q6k-gb10-decode-final.patch). It modifies
llama.cpp's `ggml/src/ggml-cuda/mmvq.cu` in four small areas.

### Exact architecture dispatch

`get_device_table_id` selects `MMVQ_PARAMETERS_BLACKWELL` on both host and device only when the
compute capability equals `GGML_CUDA_CC_DGX_SPARK`. The host calculates launch dimensions while
the compiled device specialization consumes them, so the two selectors must agree.

The equality check is intentional: the evidence covers GB10 `sm_121a`, not every GPU in the wider
Blackwell family.

### Eight-warp specialization

`calc_nwarps` returns eight only for the exact GB10 table, Q6_K weights, and one output column:

```cpp
if (table_id == MMVQ_PARAMETERS_BLACKWELL &&
        type == GGML_TYPE_Q6_K && ncols_dst == 1) {
    return 8;
}
```

One output column is the matrix-vector shape used for single-token decode. More independent Q6_K
blocks remain in flight, giving the warp scheduler useful work while other warps wait for global
memory. The change does not target multi-column prompt prefill.

### Next-block L2 prefetch

`prefetch_q6_K_block_l2` issues `prefetch.global.L2` hints at byte offsets 0 and 128, the starts of
the Q6_K `ql` and `qh` packed-weight fields. In the main `kbx` loop, lane zero calculates the block
its warp will process in the next iteration, checks the bound, and prefetches it. The fused path
does this for both the up-projection and gate-projection matrices.

The prefetch is a performance hint, not a correctness dependency. The existing demand loads and
quantized arithmetic remain unchanged. One lane issues each hint to avoid 32 redundant requests.

## 3. Inspect the isolated operation

[`src/q6k-microbench.cpp`](../src/q6k-microbench.cpp) constructs the real GGML graph without loading
the 73B model:

```text
F32 activation -> Q6_K matrix-vector multiply
               -> optional Q6_K gate multiply + fused SwiGLU
```

The key pieces are:

- `Sizes`: checked allocation accounting and a hard two-GiB safety limit;
- `RunResult`: output plus warmup/measured latency data;
- `main`: validated arguments, deterministic input construction, CPU/CUDA execution, and medians;
- `nmse`: normalized error between the CPU reference and CUDA result.

Correctness is checked before performance. Accepted prefetch outputs were also byte-identical to
the baseline GPU path because cache hints should not alter arithmetic.

## 4. Follow the validation ladder

The change moved through increasingly expensive gates:

1. compile and inspect registers, spills, SASS, and emitted prefetch instructions;
2. compare CUDA output with a CPU GGML reference;
3. run interleaved isolated baseline/candidate pairs after warmup;
4. run fresh-process full-model pairs while alternating order;
5. repeat at occupied KV-cache depths through the 7,168-token band; and
6. compile a non-GB10 control to verify the generic path remained selected.

Alternating order reduces systematic temperature, clock, and run-order bias. Medians, pair wins,
and bootstrap intervals are retained instead of reporting one favorable sample.

## 5. Know the boundary of the claim

The accepted patch is supported for exact GB10, Q6_K, and `N=1` decode. It does not change
attention, KV-cache representation, sampling, prompt prefill, other quantization formats, or other
GPU architectures. Whole-request benefit depends on the ratio of prompt processing to generated
tokens.
