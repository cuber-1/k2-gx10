# Q6_K decode long-context A/B result: initial campaign

Each independent observation is one fresh process and 128 synchronized decode
tokens over the reported band. Repetition 0 was the excluded target-depth warmup;
repetition 1 was analyzed. Positive paired change means candidate faster.

| depth band | baseline median tok/s (IQR; MAD; range) | candidate median tok/s (IQR; MAD; range) | paired median | paired IQR | wins | B-first / C-first medians | bootstrap 95% CI | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0..127 | 3.264091 (0.254988; 0.020036; 3.240244-3.512909) | 3.660003 (0.283676; 0.026055; 3.632815-3.936798) | +11.9985% | 0.3331 pp | 10/10 | +12.2111% / +11.9496% | [+11.7895%, +12.2111%] | persistent |
| 2048..2175 | 3.229841 (0.256734; 0.029197; 3.197566-3.470825) | 3.737678 (0.284085; 0.137961; 3.566540-3.917891) | +11.6166% | 0.5741 pp | 10/10 | +11.7248% / +11.5491% | [+11.3071%, +12.1152%] | persistent |
| 4096..4223 | 3.194491 (0.256548; 0.027018; 3.161857-3.435513) | 3.820274 (0.272948; 0.033400; 3.523223-3.871300) | +11.4940% | 1.4883 pp | 10/10 | +11.8275% / +11.4372% | [+11.2348%, +15.5916%] | persistent |
| 7168..7295 | 3.379342 (0.230506; 0.010772; 3.137312-3.391952) | 3.627165 (0.272403; 0.136111; 3.473919-3.768637) | +10.9621% | 0.7316 pp | 10/10 | +11.6010% / +10.7721% | [+6.8133%, +11.6010%] | persistent |

Long-context gate: **PASS**.
All-depth gate (controls conditional confirmation): **PASS**.

See `PREREGISTRATION.md`, `invocations-initial.csv`,
`paired-invocations-initial.csv`, `summary-initial.json`, raw evidence,
telemetry, and provenance for the complete record.
