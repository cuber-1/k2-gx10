# Q6_K decode long-context A/B result: repeat campaign

Each independent observation is one fresh process and 128 synchronized decode
tokens over the reported band. Repetition 0 was the excluded target-depth warmup;
repetition 1 was analyzed. Positive paired change means candidate faster.

| depth band | baseline median tok/s (IQR; MAD; range) | candidate median tok/s (IQR; MAD; range) | paired median | paired IQR | wins | B-first / C-first medians | bootstrap 95% CI | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0..127 | 3.505157 (0.008004; 0.004223; 3.499989-3.516492) | 3.921498 (0.006630; 0.003187; 3.907432-3.927652) | +11.7542% | 0.3127 pp | 10/10 | +11.7305% / +11.7780% | [+11.5729%, +11.9707%] | persistent |
| 2048..2175 | 3.461161 (0.004381; 0.002645; 3.453106-3.470630) | 3.865809 (0.006584; 0.002721; 3.852424-3.869890) | +11.5874% | 0.2817 pp | 10/10 | +11.6680% / +11.5068% | [+11.4104%, +11.7400%] | persistent |
| 4096..4223 | 3.425727 (0.008201; 0.004960; 3.416091-3.435425) | 3.823462 (0.006212; 0.002968; 3.808491-3.827192) | +11.5457% | 0.2940 pp | 10/10 | +11.4037% / +11.6566% | [+11.3421%, +11.7453%] | persistent |
| 7168..7295 | 3.370465 (0.009583; 0.005378; 3.360143-3.382716) | 3.757840 (0.010869; 0.003886; 3.742568-3.761852) | +11.2610% | 0.4246 pp | 10/10 | +11.1505% / +11.6203% | [+11.1505%, +11.6203%] | persistent |

Long-context gate: **PASS**.
All-depth gate (controls conditional confirmation): **PASS**.

See `PREREGISTRATION.md`, `invocations-repeat.csv`,
`paired-invocations-repeat.csv`, `summary-repeat.json`, raw evidence,
telemetry, and provenance for the complete record.
