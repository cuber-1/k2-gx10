# Device/resource evidence

Status: static rejection; no GPU correctness or performance run.

The exact candidate Q6 object was built only after correcting the experiment
contract. The earlier compile was terminated and its generated object deleted;
`PRECOMPILE_CORRECTION.md` records that chronology. Baseline/candidate object
and extracted-cubin hashes are in `static/sha256.txt`; the exact source patch is
`source.diff`, and the compilation entry is `static/q6-compile-command.txt`.

The exact sm_121a direct-grid symbol
`mul_mat_q<Q6_K,128,false,true>` has:

| Metric | Required | Observed |
|---|---:|---:|
| IMMA | 256 | 256 |
| FFMA | 640 | 640 |
| REG | no worse functional gate | 255 |
| STACK | 0 B | 40 B |
| LDL | 0 | 15 |
| STL | 0 | 11 |
| SHARED | accepted value | 1,024 B |

It therefore retains one full-tile math body but still spills materially. Its
inherited `tmp_fixup` parameter is declaration-only in PTX and is never loaded
or addressed. All 32 mapped generic-false main functions and all 32 fixup
functions are exact encoding/resource matches to baseline.

Because zero stack/local traffic was a predeclared hard gate, compilation did
not advance to the full library or GPU. The generated candidate object and
static artifacts are preserved; the isolated candidate source was not applied
to the accepted tree.
