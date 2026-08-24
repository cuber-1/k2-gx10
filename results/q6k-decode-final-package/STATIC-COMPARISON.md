# Static comparison

## Full sm_121a MMVQ image

Each Stage-A/final `mmvq.cu.o` contains one embedded sm_121a cubin. Both
extracted cubins have SHA-256
`33cc7e2ba2b470e358c93f47e6485c709b61375528caed3ec70434fe163a1ea8`
and `cmp` returns zero.

The following independently generated pairs are byte-identical:

- `static/stage-a/symbols.txt` and `static/final/symbols.txt`: 276 logical
  `STT_FUNC/STO_ENTRY` symbols (292 lines, 43,087 bytes including headers).
- `static/stage-a/resources.txt` and `static/final/resources.txt`: the full
  resource map (556 lines, 56,666 bytes).
- `static/stage-a/ordered-encoding.sass` and
  `static/final/ordered-encoding.sass`: all ordered raw instruction encodings
  (176,806 lines, 11,759,301 bytes).

The final normal disassembly has SHA-256
`4056825a57deb419edbf68348c77cef2ddc25204bf6d7e56dfc9dcec159b4076`
and is byte-identical to the accepted Stage-A
`exact-cubin/candidate-mmvvq.sass`.

## Exact Q6_K N=1 kernels

| Kernel | SASS SHA-256 | Registers | Shared | Stack/local | CCTL.E.PF2 | IDP.4A |
|---|---|---:|---:|---:|---:|---:|
| Nonfused | `5a5da98591ba48f11779e76f079c0511766a2959698c5af5f7c8a16b5ec0caa3` | 56 | 1920 B | 0/0 | 6 | 6 |
| Fused | `75f3096e26a9ae6d360747a2ba865870373abcd128ac290feee78d14f2f5247e` | 48 | 2816 B | 0/0 | 4 | 4 |

Both extracted kernels are byte-identical to the accepted Stage-A plain/fused
artifacts. The six static nonfused sites arise from compiler loop versioning;
one 0/128 pair executes per dynamic next-block step. The fused kernel has one
pair for the main weights and one gated pair for the gate weights.

## Compile command

After replacing the checkout root with `<SOURCE>`, the Stage-A and final nvcc
commands compare byte for byte. Their normalized response-file include lists
also compare byte for byte. The normalized command SHA-256 is
`3e2d6f0dedcecc2e28fbd21e6334e2b131319b59b5d721406a7f655fdc7a69ce`;
the normalized include-list SHA-256 is
`125ca9602fbb4efd4d5e787da7cc8896a94ba85621408edaaacd4c73b64faca8`.

## Selector proof

Source anchors in the isolated final checkout:

- Lines 85–101: the device special table requires non-HIP, non-MUSA, and
  exact `__CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK`; the fallback is generic.
- Lines 104–107: the host special table requires exact
  `cc == GGML_CUDA_CC_DGX_SPARK`.
- Lines 370–388: only special-table Q6_K N=1 selects eight warps; generic N=1
  selects four.
- Lines 614–629: the prefetch block is exact cc1210, Q6_K, N=1, one row,
  non-small-K, and bounds guarded.

The compiled Q6_K host dispatcher in
`static/final/q6k-host-selector.objdump` compares the detected cc with
`#0x4ba` (1210) and branches to the special-table path only on equality.

As a control, the exact final source was separately compiled for
compute_120a/sm_120a. Its cubin SHA-256 is
`bbb6ceea724b0b65f69e27bb0407693cdf24e6a6a4fd05fe0ea7a77cc221dd3f`.
Its complete MMVQ disassembly contains zero `CCTL.E.PF2` instructions, and its
Q6_K N=1 nonfused/fused resources are 48/46 registers and 1408/1792 bytes
shared—the generic four-warp layout rather than the cc1210 eight-warp layout.

