# Staged Nsight Compute workflow

This is a prepared, one-time privileged Stage 1 workflow. The final command is documented but was not executed while making this change.

## Stage 1 command

Run this eventually from an SSH session, not from a terminal that depends solely on GNOME:

```bash
sudo /home/dvijraicha/k2-gx10/scripts/profile_ncu_microbenchmark.sh --stage 1
```

The wrapper constructs this exact profiling shape inside a protected root-owned mode-0700 temporary directory:

```bash
/usr/local/cuda-13.0/bin/ncu \
  --target-processes application-only \
  --replay-mode kernel \
  --clock-control none \
  --kernel-name-base demangled \
  --kernel-name 'regex:^void mul_mat_q<\(ggml_type\)14, \(int\)32, \(bool\)0>\(' \
  --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --force-overwrite \
  --export '<protected-temp>/q6k-stage1-launch-occupancy' \
  /home/dvijraicha/k2-gx10/build-microbenchmark/q6k-microbench \
    --execute --allow-root-profile
```

The script wraps the profiler in `setsid` and a five-minute `timeout`, with a ten-second termination grace period. Both Nsight Compute and the fixed microbenchmark run as root. The microbenchmark permits this only because the wrapper supplies `--allow-root-profile`; ordinary root execution remains refused. `--target-processes application-only` prevents profiling child processes.

## Single authorized stage

Only Stage 1 is available. It collects `LaunchStats` and `Occupancy` into `q6k-stage1-launch-occupancy.ncu-rep`; Stage 2 and Stage 3 commands are rejected.

Stage 1 refuses pre-existing destinations. It publishes separate `.ncu-rep`, `-ncu.log`, `-raw.csv`, `-details.txt`, and `-validation.txt` files under `profiles/ncu-microbenchmark/`, all owned by `dvijraicha:dvijraicha` with mode 0600.

## Selection and replay expectation

The kernel filter selects exactly the specialization:

```text
mul_mat_q<(ggml_type)14, 32, false>
```

`--launch-count 1` limits collection to the benchmark's one matching Q6_K launch; no launch skip is used. Only `LaunchStats` and `Occupancy` are collected.

## Memory and validation

The benchmark dry-run predicts exactly 862,750,848 bytes (822.783 MiB), including a 256 MiB backend reserve. The hard limit is 2,147,483,648 bytes. Immediately before profiling, the root wrapper invokes the fixed binary as `--dry-run --allow-root-profile` and refuses if the prediction changes, reaches 2 GiB, or the hard limit changes.

After collection, the report is imported in the protected temporary directory. Validation requires exactly one distinct `(ID, Kernel Name)` result and an exact match to the type-14/J=32/non-fallback specialization before any artifact is published. Nsight Compute 2025.3.1 normalizes that specialization in imported output as `mul_mat_q<14, 32, 0>`; the launch filter retains the fully demangled cast spelling shown above.

## Security analysis

- The wrapper requires direct sudo origin from `dvijraicha`; arbitrary root invocation is refused.
- The benchmark still refuses root unless the exact `--allow-root-profile` flag is present, and rejects that flag for non-root execution.
- Only Stage 1 exists: `LaunchStats` and `Occupancy`.
- The profiler and benchmark both run as root solely for this explicitly authorized counter collection.
- The profiler and timeout run in a verified isolated session/process group; cleanup can signal only that group.
- Temporary files are root-owned mode 0700. Publication uses no-follow, exclusive-create file descriptors and assigns final ownership to UID/GID 1000 with mode 0600.
- A five-minute hard timeout bounds profiling and replay. `--clock-control none` avoids clock changes.
- No GGUF/model loader, `llama-server`, networking, arbitrary target path, GNOME control, driver setting, global counter enablement, or policy change is present.

For a non-profiling preview of paths and the exact Stage 1 shape, the script provides `--plan`; it does not invoke Nsight Compute or sudo.

