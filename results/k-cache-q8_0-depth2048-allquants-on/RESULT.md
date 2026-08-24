# K-cache Q8_0 depth-2048 result

Status: **rejected at the preregistered numerical-correctness gate; no timing
was run or cleared.**

The OFF/ON F16 control passed. The accepted-OFF process generated a frozen
128-token continuation (SHA-256
`6c8a4687eec67c33cc714a2abe1287286953a7e5cbc3460c2737d8b141498dc9`),
and the ALL_QUANTS=ON process reproduced all 129 F16 logit rows byte-for-byte.
The frozen OFF logit tensor SHA-256 is
`a3a41a5a6b528633d49f12f9feef7051125e2fc90ac4a654878ed14ea6f42d55`.

Q8_0-K/F16-V was deterministic in two fresh contexts and preserved every
argmax across all 129 rows. Distribution-sensitive metrics were comfortably
inside their gates:

- mean/max KL: `2.7009773707479431e-06` / `2.8980473583017478e-05`;
- mean/max selected-token log-probability absolute delta:
  `3.5403824155162891e-05` / `0.00035524386382235207` nat.

It nevertheless failed both preregistered raw-logit error gates:

- aggregate NMSE `0.00048069828581998947` exceeded `0.0001`;
- normalized maximum absolute difference `15.536209929563396%` exceeded `2%`.

The process loaded the model once and created four exact contexts (A1/A2 F16,
B1/B2 Q8_0), with Flash Attention, CUDA graphs, ALL_QUANTS, 81/81 GPU offload,
and requested/effective cache types logged. Peak host `MemAvailable` drop was
61,122 MiB, minimum available memory was 55,377 MiB, and peak process-group RSS
was 2,569 MiB. No contention, telemetry, memory, timeout, or fallback marker was
raised.

Primary evidence:

- parity: `runs/parity-correctness` and `artifacts/frozen`;
- Q8 gate: `runs/q8-correctness/q8-check/correctness.json`;
- per-row hashes/metrics: `runs/q8-correctness/q8-check/per-step.jsonl`;
- raw runtime/memory evidence: `runs/q8-correctness/q8-check/{stdout.txt,stderr.txt,telemetry.csv,memory-summary.txt}`;
- the pre-model GB10 telemetry-fallback attempt remains preserved at
  `runs/attempt1-gpu-memory-na`.
