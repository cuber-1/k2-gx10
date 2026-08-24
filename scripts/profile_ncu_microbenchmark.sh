#!/usr/bin/bash -p
set -euo pipefail
umask 077

readonly PROJECT_DIR="/home/dvijraicha/k2-gx10"
readonly SCRIPT_PATH="$PROJECT_DIR/scripts/profile_ncu_microbenchmark.sh"
readonly BINARY="$PROJECT_DIR/build-microbenchmark/q6k-microbench"
readonly PROFILE_DIR="$PROJECT_DIR/profiles/ncu-microbenchmark"
readonly NCU="/usr/local/cuda-13.0/bin/ncu"
readonly NCU_DOC="/opt/nvidia/nsight-compute/2025.3.1/docs/NsightComputeCli/index.html"
readonly ENV_BIN="/usr/bin/env"
readonly TIMEOUT="/usr/bin/timeout"
readonly SETSID="/usr/bin/setsid"
readonly PS="/usr/bin/ps"
readonly READLINK="/usr/bin/readlink"
readonly STAT="/usr/bin/stat"
readonly ID="/usr/bin/id"
readonly MKTEMP="/usr/bin/mktemp"
readonly RM="/usr/bin/rm"
readonly SHA256SUM="/usr/bin/sha256sum"
readonly PYTHON="/usr/bin/python3.12"
readonly SUDO="/usr/bin/sudo"

readonly TARGET_USER="dvijraicha"
readonly TARGET_HOME="/home/dvijraicha"
readonly HARD_LIMIT_BYTES=2147483648
readonly PROFILE_TIMEOUT_SECONDS=300
readonly IMPORT_TIMEOUT_SECONDS=60
readonly KILL_GRACE_SECONDS=10
readonly SAFE_TARGET_PATH="/usr/local/cuda-13.0/bin:/usr/bin:/bin"

MATMUL_COLUMNS=25
PROFILE_MODE="mmq"
KERNEL_J=""
EXPECTED_GUARDED_BYTES=""
KERNEL_FILTER=""
KERNEL_NAME_RE=""

STAGE=""
STAGE_LABEL=""
PLAN_ONLY=false
TARGET_UID=""
TARGET_GID=""
TEMP_DIR=""
TEMP_BASE=""
TEMP_REPORT=""
TEMP_LOG=""
TEMP_CSV=""
TEMP_DETAILS=""
TEMP_VALIDATION=""
DEST_BASE=""
PROFILE_PID=""
PROFILE_PGID=""
SCRIPT_PGID=""
SECTIONS=()

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_regular_executable() {
  local path="$1"
  [[ "$path" == /* && -f "$path" && ! -L "$path" && -x "$path" ]] ||
    die "required executable is missing, non-regular, symlinked, or not executable: $path"
}

read_process_field() {
  local value
  value="$($PS -o "$1=" -p "$2")" || return 1
  value="${value//[[:space:]]/}"
  [[ -n "$value" ]] || return 1
  printf '%s\n' "$value"
}

stop_profile_group() {
  local current_pgid current_sid
  [[ -n "$PROFILE_PID" && -n "$PROFILE_PGID" ]] || return 0
  [[ "$PROFILE_PID" =~ ^[0-9]+$ && "$PROFILE_PGID" =~ ^[0-9]+$ ]] || return 1
  (( PROFILE_PID > 1 && PROFILE_PGID > 1 )) || return 1
  [[ "$PROFILE_PGID" != "$SCRIPT_PGID" ]] || return 1
  current_pgid="$(read_process_field pgid "$PROFILE_PID")" || return 0
  current_sid="$(read_process_field sid "$PROFILE_PID")" || return 0
  [[ "$current_pgid" == "$PROFILE_PGID" && "$current_sid" == "$PROFILE_PGID" ]] || return 1
  kill -TERM -- "-$PROFILE_PGID" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_profile_group || status=1
  if [[ -n "$PROFILE_PID" ]]; then wait "$PROFILE_PID" 2>/dev/null || true; fi
  if [[ -n "$TEMP_DIR" ]]; then
    if [[ "$TEMP_DIR" == /tmp/q6k-ncu-microbenchmark.* && -d "$TEMP_DIR" && ! -L "$TEMP_DIR" && "$($STAT -c '%u:%a' "$TEMP_DIR")" == "0:700" ]]; then
      "$RM" -rf --one-file-system -- "$TEMP_DIR"
    else
      printf 'ERROR: refusing unsafe temporary-directory cleanup: %s\n' "$TEMP_DIR" >&2
      status=1
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
  cat <<EOF
Usage:
  $SCRIPT_PATH --plan [--columns 25|100|512|1024 | --decode-fused]
  sudo $SCRIPT_PATH --stage 1
  sudo $SCRIPT_PATH --stage 2 [--columns 25|100|512|1024]
  sudo $SCRIPT_PATH --stage 1 --decode-fused
  sudo $SCRIPT_PATH --stage 2 --decode-fused

No profiling occurs with --plan. Each stage refuses pre-existing destinations.
Stage 1 remains fixed at 25 columns; Stage 2 accepts only the reviewed shapes.
EOF
}

configure_shape() {
  if [[ "$PROFILE_MODE" == "decode-fused" ]]; then
    MATMUL_COLUMNS=1
    EXPECTED_GUARDED_BYTES=1234969888
    KERNEL_FILTER="regex:^void mul_mat_vec_q<\\(ggml_type\\)14, \\(int\\)1, \\(bool\\)1, \\(bool\\)0>\\("
    KERNEL_NAME_RE="^void mul_mat_vec_q<14, 1, 1, 0>\\(.*\\)$"
    return
  fi
  case "$MATMUL_COLUMNS" in
    25)  KERNEL_J=32;  EXPECTED_GUARDED_BYTES=862750848 ;;
    100) KERNEL_J=112; EXPECTED_GUARDED_BYTES=905224128 ;;
    512)  KERNEL_J=128; EXPECTED_GUARDED_BYTES=1138528768 ;;
    1024) KERNEL_J=128; EXPECTED_GUARDED_BYTES=1428460032 ;;
    *) die "--columns must be one of: 25, 100, 512, 1024" ;;
  esac
  KERNEL_FILTER="regex:^void mul_mat_q<\\(ggml_type\\)14, \\(int\\)${KERNEL_J}, \\(bool\\)0>\\("
  KERNEL_NAME_RE="^void mul_mat_q<14, ${KERNEL_J}, 0>\\(.*\\)$"
}

parse_args() {
  case "${1:-}" in
    --plan)
      (( $# == 1 || $# == 2 || $# == 3 )) || die "invalid --plan arguments"
      if (( $# == 2 )); then
        [[ "$2" == "--decode-fused" ]] || die "expected --decode-fused after --plan"
        PROFILE_MODE="decode-fused"
      elif (( $# == 3 )); then
        [[ "$2" == "--columns" ]] || die "expected --columns after --plan"
        MATMUL_COLUMNS="$3"
      fi
      PLAN_ONLY=true
      configure_shape
      return
      ;;
    --stage)
      (( $# == 2 || $# == 3 || $# == 4 )) || die "invalid stage arguments"
      [[ "$2" == "1" || "$2" == "2" ]] || die "stage must be 1 or 2"
      STAGE="$2"
      if (( $# == 3 )); then
        [[ "$3" == "--decode-fused" ]] || die "expected --decode-fused after stage"
        PROFILE_MODE="decode-fused"
      elif (( $# == 4 )); then
        [[ "$3" == "--columns" ]] || die "expected --columns after stage"
        MATMUL_COLUMNS="$4"
      fi
      configure_shape
      if [[ "$STAGE" == "1" ]]; then
        [[ "$PROFILE_MODE" == "decode-fused" || "$MATMUL_COLUMNS" == "25" ]] || die "MMQ Stage 1 is fixed at 25 columns"
        if [[ "$PROFILE_MODE" == "decode-fused" ]]; then
          STAGE_LABEL="decode-stage1-launch-occupancy"
        else
          STAGE_LABEL="stage1-launch-occupancy"
        fi
        SECTIONS=(--section LaunchStats --section Occupancy)
      else
        if [[ "$PROFILE_MODE" == "decode-fused" ]]; then
          STAGE_LABEL="decode-stage2-bottleneck-analysis"
        elif [[ "$MATMUL_COLUMNS" == "25" ]]; then
          STAGE_LABEL="stage2-bottleneck-analysis"
        else
          STAGE_LABEL="stage2-bottleneck-analysis-n${MATMUL_COLUMNS}"
        fi
        SECTIONS=(
          --section SpeedOfLight
          --section ComputeWorkloadAnalysis
          --section MemoryWorkloadAnalysis
          --section WarpStateStats
          --section SourceCounters
          --section LaunchStats
          --section Occupancy
        )
      fi
      DEST_BASE="$PROFILE_DIR/q6k-$STAGE_LABEL"
      ;;
    *) usage >&2; exit 2 ;;
  esac
}

print_plan() {
  local stage1_label="stage1-launch-occupancy"
  local stage1_arg=""
  local target_decode_arg=""
  if [[ "$PROFILE_MODE" == "decode-fused" ]]; then
    stage1_label="decode-stage1-launch-occupancy"
    stage1_arg=" --decode-fused"
    target_decode_arg=" --fused-swiglu"
  fi
  cat <<EOF
Installed Nsight Compute: 2025.3.1 (confirmed from installed path and documentation)
Target process behavior: --target-processes application-only; ncu launches the fixed benchmark directly as root.
Kernel filter: $KERNEL_FILTER
Launch selection: --launch-count 1; no launch skip
Profile mode: $PROFILE_MODE
Columns: $MATMUL_COLUMNS
Predicted guarded allocation: $EXPECTED_GUARDED_BYTES bytes
Hard refusal limit: $HARD_LIMIT_BYTES bytes (2048 MiB)
Stage 1 expected collection: one selected kernel result; normally one metric pass and zero additional
  kernel replays, subject to the device's counter-pass scheduler. The profiler log is authoritative.
Stage 1 report: $PROFILE_DIR/q6k-$stage1_label.ncu-rep

Exact Stage 1 command:
  sudo $SCRIPT_PATH --stage 1$stage1_arg

The wrapper's exact profiler/target shape is:
  $NCU --target-processes application-only \\
    --replay-mode kernel --clock-control none --kernel-name-base demangled \\
    --kernel-name '$KERNEL_FILTER' --launch-count 1 \\
    --section LaunchStats --section Occupancy --export '<protected-temp>/q6k-$stage1_label' \\
    $BINARY --execute --columns $MATMUL_COLUMNS$target_decode_arg --allow-root-profile

Recommendation: run this one-time stage from SSH so profiler output and recovery visibility remain available
if the graphical session becomes unhealthy. Do not modify drivers, counters, clocks, or system policy.
EOF
}

validate_installed_options() {
  [[ -f "$NCU_DOC" && ! -L "$NCU_DOC" && -r "$NCU_DOC" ]] || die "installed 2025.3.1 CLI documentation is unavailable"
  grep -q 'application-only' "$NCU_DOC" || die "installed documentation lacks target-processes=application-only"
  grep -q 'launch-count' "$NCU_DOC" || die "installed documentation lacks launch-count"
}

validate_fixed_paths() {
  local path binary_owner
  for path in "$NCU" "$ENV_BIN" "$TIMEOUT" "$SETSID" "$PS" "$READLINK" "$STAT" "$ID" "$MKTEMP" "$RM" "$SHA256SUM" "$PYTHON" "$SUDO" "$BINARY"; do
    require_regular_executable "$path"
  done
  [[ -d "$PROJECT_DIR" && ! -L "$PROJECT_DIR" ]] || die "unsafe project directory"
  [[ -d "$PROFILE_DIR" && ! -L "$PROFILE_DIR" ]] || die "unsafe profile directory"
  TARGET_UID="$($ID -u "$TARGET_USER")" || die "target user does not exist"
  TARGET_GID="$($ID -g "$TARGET_USER")" || die "cannot resolve target group"
  [[ "$TARGET_UID" == "1000" && "$TARGET_GID" == "1000" ]] || die "unexpected target UID/GID"
  [[ -d "$TARGET_HOME" && ! -L "$TARGET_HOME" ]] || die "unexpected target home"
  [[ "$("$STAT" -c '%u:%g' "$TARGET_HOME")" == "$TARGET_UID:$TARGET_GID" ]] || die "target home ownership mismatch"
  binary_owner="$($STAT -c '%U:%G' "$BINARY")"
  [[ "$binary_owner" == "$TARGET_USER:$TARGET_USER" ]] || die "benchmark is not owned by $TARGET_USER"
  [[ "$($STAT -c '%u:%g' "$PROFILE_DIR")" == "$TARGET_UID:$TARGET_GID" ]] || die "profile directory ownership mismatch"
  validate_installed_options
}

validate_sudo_origin() {
  local parent
  (( EUID == 0 )) || die "profiling stages must be invoked through sudo; use --plan without sudo"
  [[ ${SUDO_UID+x} && ${SUDO_GID+x} && ${SUDO_USER+x} && ${SUDO_COMMAND+x} ]] || die "root execution lacks sudo origin"
  [[ "$SUDO_USER" == "$TARGET_USER" && "$SUDO_UID" == "$TARGET_UID" && "$SUDO_GID" == "$TARGET_GID" ]] ||
    die "sudo origin is not exactly $TARGET_USER"
  parent="$($READLINK -f "/proc/$PPID/exe")" || die "cannot inspect sudo parent"
  [[ "$parent" == "$SUDO" ]] || die "script was not started directly by sudo"
}


validate_destinations_absent() {
  local suffix path
  for suffix in .ncu-rep -ncu.log -raw.csv -details.txt -validation.txt; do
    path="$DEST_BASE$suffix"
    [[ ! -e "$path" && ! -L "$path" ]] || die "refusing pre-existing destination: $path"
  done
}

create_temp_dir() {
  TEMP_DIR="$($MKTEMP -d --tmpdir=/tmp q6k-ncu-microbenchmark.XXXXXXXX)" || die "mktemp failed"
  [[ "$TEMP_DIR" == /tmp/q6k-ncu-microbenchmark.* && -d "$TEMP_DIR" && ! -L "$TEMP_DIR" ]] || die "unsafe temporary path"
  [[ "$($STAT -c '%u:%a' "$TEMP_DIR")" == "0:700" ]] || die "temporary directory is not root-owned mode 0700"
  mkdir -m 700 "$TEMP_DIR/home" "$TEMP_DIR/tmp"
  TEMP_BASE="$TEMP_DIR/q6k-$STAGE_LABEL"
  TEMP_REPORT="$TEMP_BASE.ncu-rep"
  TEMP_LOG="$TEMP_BASE-ncu.log"
  TEMP_CSV="$TEMP_BASE-raw.csv"
  TEMP_DETAILS="$TEMP_BASE-details.txt"
  TEMP_VALIDATION="$TEMP_BASE-validation.txt"
}

validate_allocation() {
  local dry_output guarded hard
  local target_args=(--dry-run --columns "$MATMUL_COLUMNS" --allow-root-profile)
  if [[ "$PROFILE_MODE" == "decode-fused" ]]; then target_args+=(--fused-swiglu); fi
  dry_output="$("$BINARY" "${target_args[@]}")" ||
    die "benchmark dry-run gate failed"
  grep -q '^DRY RUN PASS:' <<<"$dry_output" || die "benchmark dry-run did not reach its safe exit"
  guarded="$(sed -n 's/^  GUARDED TOTAL[[:space:]]*\([0-9][0-9]*\) B.*/\1/p' <<<"$dry_output")"
  hard="$(sed -n 's/^  HARD LIMIT[[:space:]]*\([0-9][0-9]*\) B.*/\1/p' <<<"$dry_output")"
  [[ "$guarded" =~ ^[0-9]+$ && "$hard" =~ ^[0-9]+$ ]] || die "cannot parse allocation gate"
  (( guarded < HARD_LIMIT_BYTES && hard == HARD_LIMIT_BYTES )) || die "predicted benchmark allocation is not below 2 GiB"
  [[ "$guarded" == "$EXPECTED_GUARDED_BYTES" ]] ||
    die "predicted allocation changed for columns=$MATMUL_COLUMNS: expected $EXPECTED_GUARDED_BYTES bytes, got $guarded bytes"
  printf 'Allocation gate: %s bytes guarded, %s bytes hard limit\n' "$guarded" "$hard"
}

validate_import_csv() {
  "$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="/usr/bin:/bin" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
    "$PYTHON" -I -c '
import csv, re, sys
path, expected = sys.argv[1:]
with open(path, newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(line for line in stream if not line.startswith("==")))
if not rows:
    raise SystemExit("ncu CSV import contained no metric rows")
required = {"ID", "Kernel Name"}
if not required.issubset(rows[0]):
    raise SystemExit("ncu CSV import lacks ID or Kernel Name columns")
results = {(row["ID"], row["Kernel Name"]) for row in rows if row.get("ID") and row.get("Kernel Name")}
if len(results) != 1:
    raise SystemExit(f"expected exactly one kernel result, found {len(results)}")
kernel_id, name = next(iter(results))
if re.fullmatch(expected, name) is None:
    raise SystemExit(f"unexpected kernel name: {name}")
print("status=validated")
print(f"result_count=1")
print(f"kernel_id={kernel_id}")
print(f"kernel_name={name}")
' "$TEMP_CSV" "$KERNEL_NAME_RE" >"$TEMP_VALIDATION"
}

publish_file() {
  local source="$1" destination="$2" destination_name
  destination_name="${destination##*/}"
  [[ -f "$source" && ! -L "$source" && -s "$source" ]] || die "unsafe staged artifact: $source"
  [[ "$destination" == "$PROFILE_DIR/"* && ! -e "$destination" && ! -L "$destination" ]] || die "unsafe destination: $destination"
  "$ENV_BIN" -i PATH="/usr/bin:/bin" LC_ALL=C "$PYTHON" -I -c '
import os, stat, sys
source, output_dir, name, uid_text, gid_text = sys.argv[1:]
uid, gid = int(uid_text), int(gid_text)
if not name or "/" in name or name in {".", ".."}: raise SystemExit("unsafe destination name")
dfd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
sfd = dfd_out = None
created = False
try:
    ds = os.fstat(dfd)
    if (ds.st_uid, ds.st_gid) != (uid, gid): raise SystemExit("output ownership changed")
    sfd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    ss = os.fstat(sfd)
    if not stat.S_ISREG(ss.st_mode) or ss.st_uid != 0 or ss.st_size <= 0: raise SystemExit("unsafe source")
    dfd_out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
    created = True
    while True:
        block = os.read(sfd, 1024 * 1024)
        if not block: break
        view = memoryview(block)
        while view: view = view[os.write(dfd_out, view):]
    os.fchmod(dfd_out, 0o600)
    os.fchown(dfd_out, uid, gid)
    os.fsync(dfd_out)
except BaseException:
    if dfd_out is not None: os.close(dfd_out); dfd_out = None
    if created:
        try: os.unlink(name, dir_fd=dfd)
        except FileNotFoundError: pass
    raise
finally:
    if dfd_out is not None: os.close(dfd_out)
    if sfd is not None: os.close(sfd)
    os.close(dfd)
' "$source" "$PROFILE_DIR" "$destination_name" "$TARGET_UID" "$TARGET_GID"
}

run_stage() {
  local status suffix
  local target_command=(
    "$BINARY" --execute --columns "$MATMUL_COLUMNS" --allow-root-profile
  )
  if [[ "$PROFILE_MODE" == "decode-fused" ]]; then target_command+=(--fused-swiglu); fi
  local ncu_command=(
    "$NCU"
    --target-processes application-only
    --replay-mode kernel
    --clock-control none
    --kernel-name-base demangled
    --kernel-name "$KERNEL_FILTER"
    --launch-count 1
    "${SECTIONS[@]}"
    --force-overwrite
    --export "$TEMP_BASE"
    "${target_command[@]}"
  )

  printf 'Stage %s command:' "$STAGE"; printf ' %q' "${ncu_command[@]}"; printf '\n'
  printf 'Protected staging: %s\nFinal report: %s.ncu-rep\n' "$TEMP_DIR" "$DEST_BASE"

  "$SETSID" "$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="$SAFE_TARGET_PATH" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
    "$TIMEOUT" --signal=TERM --kill-after="${KILL_GRACE_SECONDS}s" "${PROFILE_TIMEOUT_SECONDS}s" \
    "${ncu_command[@]}" >"$TEMP_LOG" 2>&1 &
  PROFILE_PID=$!
  PROFILE_PGID="$(read_process_field pgid "$PROFILE_PID")" || die "cannot determine profiler process group"
  [[ "$PROFILE_PGID" == "$PROFILE_PID" ]] || die "profiler timeout is not its process-group leader"
  [[ "$(read_process_field sid "$PROFILE_PID")" == "$PROFILE_PGID" ]] || die "profiler is not in an isolated session"

  if wait "$PROFILE_PID"; then
    status=0
  else
    status=$?
  fi
  PROFILE_PID=""; PROFILE_PGID=""
  if (( status != 0 )); then
    sed -n '1,260p' "$TEMP_LOG" >&2
    die "Stage $STAGE profiler failed or timed out with status $status"
  fi
  if [[ ! -f "$TEMP_REPORT" || -L "$TEMP_REPORT" || ! -s "$TEMP_REPORT" ]]; then
    sed -n '1,260p' "$TEMP_LOG" >&2
    die "ncu did not create a report"
  fi
  "$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="$SAFE_TARGET_PATH" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
    "$TIMEOUT" --signal=TERM --kill-after=5s "${IMPORT_TIMEOUT_SECONDS}s" \
    "$NCU" --import "$TEMP_REPORT" --csv --page raw --print-kernel-base demangled >"$TEMP_CSV"
  [[ -s "$TEMP_CSV" ]] || die "ncu CSV import was empty"
  validate_import_csv
  {
    "$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="$SAFE_TARGET_PATH" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
      "$TIMEOUT" --signal=TERM --kill-after=5s "${IMPORT_TIMEOUT_SECONDS}s" \
      "$NCU" --import "$TEMP_REPORT" --page details --print-details all --print-kernel-base demangled
  } >"$TEMP_DETAILS"
  printf 'stage=%s\nsections=' "$STAGE" >>"$TEMP_VALIDATION"
  printf '%s ' "${SECTIONS[@]}" >>"$TEMP_VALIDATION"
  printf '\ncolumns=%s\n' "$MATMUL_COLUMNS" >>"$TEMP_VALIDATION"
  printf 'profile_mode=%s\n' "$PROFILE_MODE" >>"$TEMP_VALIDATION"
  printf '\nallocation_bytes=%s\nreport_sha256=' "$EXPECTED_GUARDED_BYTES" >>"$TEMP_VALIDATION"
  "$SHA256SUM" "$TEMP_REPORT" | awk '{print $1}' >>"$TEMP_VALIDATION"

  publish_file "$TEMP_REPORT" "$DEST_BASE.ncu-rep"
  publish_file "$TEMP_LOG" "$DEST_BASE-ncu.log"
  publish_file "$TEMP_CSV" "$DEST_BASE-raw.csv"
  publish_file "$TEMP_DETAILS" "$DEST_BASE-details.txt"
  publish_file "$TEMP_VALIDATION" "$DEST_BASE-validation.txt"
  for suffix in .ncu-rep -ncu.log -raw.csv -details.txt -validation.txt; do
    [[ "$($STAT -c '%u:%g:%a' "$DEST_BASE$suffix")" == "$TARGET_UID:$TARGET_GID:600" ]] ||
      die "published ownership or mode is wrong: $DEST_BASE$suffix"
  done
  printf 'Stage %s validated and published under %s\n' "$STAGE" "$PROFILE_DIR"
}

parse_args "$@"
validate_fixed_paths
if [[ "$PLAN_ONLY" == true ]]; then
  print_plan
  exit 0
fi
validate_sudo_origin
validate_destinations_absent
SCRIPT_PGID="$(read_process_field pgid "$$")" || die "cannot determine script process group"
create_temp_dir
validate_allocation
run_stage
