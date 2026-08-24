#!/usr/bin/bash -p
printf "ERROR: historical full-model profiler is disabled; use docs/safe-ncu-microbenchmark.md\n" >&2
exit 64

set -euo pipefail
umask 077

# Pinned to this host: never resolve executables or inputs through caller state.
readonly PROJECT_DIR="/home/dvijraicha/k2-gx10"
readonly SCRIPT_PATH="/home/dvijraicha/k2-gx10/scripts/profile_ncu.sh"
readonly CLIENT="/home/dvijraicha/k2-gx10/client_test.py"
readonly SERVER="/home/dvijraicha/llama.cpp/build/bin/llama-server"
readonly MODEL_DIR="/home/dvijraicha/.cache/huggingface/hub/models--benjaminradio--K2-Think-V2-GGUF/snapshots/3064ec56b7c735f4f133aa10cfcca3ef3bd718f7"
readonly MODEL_1="$MODEL_DIR/K2-Think-V2-Q6_K-00001-of-00004.gguf"
readonly MODEL_2="$MODEL_DIR/K2-Think-V2-Q6_K-00002-of-00004.gguf"
readonly MODEL_3="$MODEL_DIR/K2-Think-V2-Q6_K-00003-of-00004.gguf"
readonly MODEL_4="$MODEL_DIR/K2-Think-V2-Q6_K-00004-of-00004.gguf"
readonly MODEL_1_TARGET="/home/dvijraicha/.cache/huggingface/hub/models--benjaminradio--K2-Think-V2-GGUF/blobs/c30e6f1322a590c5772ae606e4ff36677a3685d2c7fa335e21611bb88790e0d0"
readonly MODEL_2_TARGET="/home/dvijraicha/.cache/huggingface/hub/models--benjaminradio--K2-Think-V2-GGUF/blobs/c6f36e13321422d9c4cc20b5160de70afede5599a1deced1cdbe13c515cddaa1"
readonly MODEL_3_TARGET="/home/dvijraicha/.cache/huggingface/hub/models--benjaminradio--K2-Think-V2-GGUF/blobs/674aff17e64716b92ebf6bb491e05144522f66d91153ba75a41aa95e6154ba47"
readonly MODEL_4_TARGET="/home/dvijraicha/.cache/huggingface/hub/models--benjaminradio--K2-Think-V2-GGUF/blobs/88ea2b5a08bf06431430550d04a6793ce99dcde028951f2cd9221671d688d96a"

readonly NCU="/usr/local/cuda-13.0/bin/ncu"
readonly CURL="/usr/bin/curl"
readonly SS="/usr/bin/ss"
readonly SETSID="/usr/bin/setsid"
readonly PS="/usr/bin/ps"
readonly SUDO="/usr/bin/sudo"
readonly ID="/usr/bin/id"
readonly STAT="/usr/bin/stat"
readonly DATE="/usr/bin/date"
readonly MKTEMP="/usr/bin/mktemp"
readonly MKDIR="/usr/bin/mkdir"
readonly RM="/usr/bin/rm"
readonly SED="/usr/bin/sed"
readonly READLINK="/usr/bin/readlink"
readonly PYTHON="/usr/bin/python3.12"
readonly ENV_BIN="/usr/bin/env"
readonly BASH_BIN="/usr/bin/bash"
readonly SLEEP="/usr/bin/sleep"

readonly PROFILE_PARENT="$PROJECT_DIR/profiles"
readonly PROFILE_DIR="$PROFILE_PARENT/ncu"
readonly HOST="127.0.0.1"
readonly PORT="30000"
readonly REASONING_BUDGET="256"
readonly MAX_TOKENS="320"
readonly LAUNCH_SKIP="32"
readonly LAUNCH_COUNT="1"
readonly KERNEL_FILTER='regex:^void mul_mat_q<\(ggml_type\)14, \(int\)32, \(bool\)0>'
readonly KERNEL_NAME_RE='^void mul_mat_q<\(ggml_type\)14, \(int\)32, \(bool\)0>.*$'
readonly SAFE_CHILD_PATH="/usr/local/cuda-13.0/bin:/usr/bin:/bin"

export PATH="/usr/bin:/bin"
export HOME="/root"
export LC_ALL="C"
IFS=$' \t\n'
unset BASH_ENV CDPATH ENV GLOBIGNORE PYTHONHOME PYTHONPATH

CHECK_ONLY=false
RUN_ID=""
REPORT_BASE=""
REPORT_FILE=""
TEXT_REPORT=""
NCU_LOG=""
REQUEST_LOG=""
TEMP_DIR=""
TEMP_REPORT_BASE=""
TEMP_REPORT_FILE=""
TEMP_TEXT_REPORT=""
TEMP_CSV_REPORT=""
TEMP_NCU_LOG=""
TEMP_REQUEST_LOG=""
GATE_FILE=""
PID_FILE=""
NCU_PID=""
LAUNCHER_PID=""
PROFILE_PGID=""
SCRIPT_PGID=""
TARGET_UID=""
TARGET_GID=""

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

require_absolute() {
  [[ "$1" == /* ]] || die "path is not absolute: $1"
}

require_executable() {
  require_absolute "$1"
  [[ -f "$1" && ! -L "$1" && -x "$1" ]] || die "required executable is not a regular executable file: $1"
}

reject_symlink_components() {
  local path="$1"
  local rest component current=""

  require_absolute "$path"
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then
      component="${rest%%/*}"
      rest="${rest#*/}"
    else
      component="$rest"
      rest=""
    fi
    [[ -n "$component" ]] || continue
    current="$current/$component"
    [[ ! -L "$current" ]] || die "symlink is forbidden in output path: $current"
  done
}

validate_run_id() {
  local value="$1"
  [[ ${#value} -ge 1 && ${#value} -le 64 ]] || die "NCU_RUN_ID must contain 1-64 characters"
  [[ "$value" =~ ^[A-Za-z0-9_-]+$ ]] || die "NCU_RUN_ID may contain only ASCII letters, digits, hyphens, and underscores"
}

validate_model_shard() {
  local path="$1"
  local expected="$2"
  local resolved

  require_absolute "$path"
  require_absolute "$expected"
  [[ -e "$path" && -r "$path" ]] || die "model shard is missing or unreadable: $path"
  resolved="$($READLINK -f -- "$path")" || die "cannot resolve model shard: $path"
  [[ "$resolved" == "$expected" ]] || die "model shard resolves to an unexpected file: $path"
  [[ -f "$resolved" && ! -L "$resolved" && -r "$resolved" ]] || die "resolved model shard is unsafe: $resolved"
}

read_owner() {
  "$STAT" -c '%u:%g' -- "$1" || die "cannot read ownership: $1"
}

validate_fixed_paths() {
  local executable repo_owner

  require_absolute "$PROJECT_DIR"
  [[ -d "$PROJECT_DIR" && ! -L "$PROJECT_DIR" ]] || die "repository root is not a real directory: $PROJECT_DIR"

  for executable in \
    "$NCU" "$CURL" "$SS" "$SETSID" "$PS" "$SUDO" "$ID" "$STAT" "$DATE" \
    "$MKTEMP" "$MKDIR" "$RM" "$SED" \
    "$READLINK" "$PYTHON" "$ENV_BIN" "$BASH_BIN" "$SLEEP" "$SERVER"; do
    require_executable "$executable"
  done

  require_absolute "$CLIENT"
  [[ -f "$CLIENT" && ! -L "$CLIENT" && -r "$CLIENT" ]] || die "client is not a safe regular file: $CLIENT"
  validate_model_shard "$MODEL_1" "$MODEL_1_TARGET"
  validate_model_shard "$MODEL_2" "$MODEL_2_TARGET"
  validate_model_shard "$MODEL_3" "$MODEL_3_TARGET"
  validate_model_shard "$MODEL_4" "$MODEL_4_TARGET"

  reject_symlink_components "$PROFILE_PARENT"
  reject_symlink_components "$PROFILE_DIR"
  [[ -d "$PROFILE_PARENT" ]] || die "profile parent does not exist: $PROFILE_PARENT"
  [[ -d "$PROFILE_DIR" ]] || die "profile directory does not exist: $PROFILE_DIR"
  repo_owner="$(read_owner "$PROJECT_DIR")"
  [[ "$(read_owner "$PROFILE_PARENT")" == "$repo_owner" ]] || die "profile parent ownership does not match repository ownership"
  [[ "$(read_owner "$PROFILE_DIR")" == "$repo_owner" ]] || die "profile directory ownership does not match repository ownership"

  [[ "$HOST" == "127.0.0.1" ]] || die "server host is not loopback"
  [[ "$KERNEL_FILTER" == 'regex:^void mul_mat_q<\(ggml_type\)14, \(int\)32, \(bool\)0>' ]] || die "kernel filter changed unexpectedly"
  [[ "$LAUNCH_SKIP" == "32" && "$LAUNCH_COUNT" == "1" ]] || die "launch bounds changed unexpectedly"
}

set_output_paths() {
  REPORT_BASE="$PROFILE_DIR/k2-q6k-mmq-$RUN_ID"
  REPORT_FILE="$REPORT_BASE.ncu-rep"
  TEXT_REPORT="$REPORT_BASE-report.txt"
  NCU_LOG="$REPORT_BASE-ncu.log"
  REQUEST_LOG="$REPORT_BASE-request.txt"
  reject_symlink_components "$REPORT_FILE"
  reject_symlink_components "$TEXT_REPORT"
  reject_symlink_components "$NCU_LOG"
  reject_symlink_components "$REQUEST_LOG"
}

validate_destinations_absent() {
  local path
  for path in "$REPORT_FILE" "$TEXT_REPORT" "$NCU_LOG" "$REQUEST_LOG"; do
    [[ ! -e "$path" && ! -L "$path" ]] || die "refusing pre-existing output path: $path"
  done
}

validate_sudo_origin() {
  local repo_uid repo_gid sudo_parent sudo_user_uid sudo_user_gid

  [[ $EUID -eq 0 ]] || die "profiling mode must be invoked with sudo; use --check-only for an unprivileged validation"
  [[ ${SUDO_UID+x} && ${SUDO_GID+x} && ${SUDO_USER+x} && ${SUDO_COMMAND+x} ]] || die "root execution requires complete sudo-origin information"
  [[ "$SUDO_COMMAND" == "$SCRIPT_PATH" ]] || die "SUDO_COMMAND is not the exact approved script path"
  sudo_parent="$($READLINK -f -- "/proc/$PPID/exe")" || die "cannot verify the sudo parent process"
  [[ "$sudo_parent" == "$SUDO" ]] || die "root execution was not started directly by $SUDO"
  is_uint "$SUDO_UID" && is_uint "$SUDO_GID" || die "SUDO_UID and SUDO_GID must be decimal integers"
  (( 10#$SUDO_UID > 0 && 10#$SUDO_GID > 0 )) || die "SUDO_UID and SUDO_GID must be nonzero"
  sudo_user_uid="$($ID -u -- "$SUDO_USER")" || die "SUDO_USER does not identify a local user"
  sudo_user_gid="$($ID -g -- "$SUDO_USER")" || die "cannot determine SUDO_USER's primary group"
  [[ "$sudo_user_uid" == "$SUDO_UID" && "$sudo_user_gid" == "$SUDO_GID" ]] || die "sudo user identity does not match SUDO_UID/SUDO_GID"
  repo_uid="$($STAT -c '%u' -- "$PROJECT_DIR")"
  repo_gid="$($STAT -c '%g' -- "$PROJECT_DIR")"
  [[ "$SUDO_UID" == "$repo_uid" ]] || die "SUDO_UID does not own $PROJECT_DIR"
  [[ "$SUDO_GID" == "$repo_gid" ]] || die "SUDO_GID does not match the group of $PROJECT_DIR"
  TARGET_UID="$SUDO_UID"
  TARGET_GID="$SUDO_GID"
}

read_process_field() {
  local value
  value="$($PS -o "$1=" -p "$2")" || return 1
  value="${value//[[:space:]]/}"
  [[ -n "$value" ]] || return 1
  printf '%s\n' "$value"
}

validate_profile_group() {
  local current_pgid current_sid
  is_uint "$NCU_PID" && (( NCU_PID > 1 )) || die "invalid profiler PID"
  is_uint "$PROFILE_PGID" && (( PROFILE_PGID > 1 )) || die "invalid profiler process group"
  is_uint "$SCRIPT_PGID" && (( SCRIPT_PGID > 1 )) || die "invalid script process group"
  [[ "$PROFILE_PGID" != "$SCRIPT_PGID" ]] || die "profiler process group is the script's process group"
  current_pgid="$(read_process_field pgid "$NCU_PID")" || die "cannot verify profiler process group"
  current_sid="$(read_process_field sid "$NCU_PID")" || die "cannot verify profiler session"
  [[ "$current_pgid" == "$PROFILE_PGID" ]] || die "profiler process group changed"
  [[ "$current_sid" == "$PROFILE_PGID" ]] || die "profiler is not the leader of its isolated session"
}

stop_profile_group() {
  local current_pgid current_sid
  [[ -n "$NCU_PID" && -n "$PROFILE_PGID" ]] || return 0
  if ! is_uint "$NCU_PID" || ! is_uint "$PROFILE_PGID" || (( NCU_PID <= 1 || PROFILE_PGID <= 1 )); then
    printf 'ERROR: refusing unsafe cleanup PID/PGID\n' >&2
    return 1
  fi
  if [[ -z "$SCRIPT_PGID" || "$PROFILE_PGID" == "$SCRIPT_PGID" ]]; then
    printf 'ERROR: refusing to signal the script process group\n' >&2
    return 1
  fi
  current_pgid="$(read_process_field pgid "$NCU_PID")" || return 0
  current_sid="$(read_process_field sid "$NCU_PID")" || return 0
  if [[ "$current_pgid" != "$PROFILE_PGID" || "$current_sid" != "$PROFILE_PGID" ]]; then
    printf 'ERROR: refusing to signal an unverified process group\n' >&2
    return 1
  fi
  kill -INT -- "-$PROFILE_PGID" 2>/dev/null || true
}

cleanup_temp_dir() {
  local owner
  [[ -n "$TEMP_DIR" ]] || return 0
  [[ "$TEMP_DIR" == /tmp/k2-ncu-profile.* ]] || {
    printf 'ERROR: refusing to remove unexpected temporary path: %s\n' "$TEMP_DIR" >&2
    return 1
  }
  [[ -d "$TEMP_DIR" && ! -L "$TEMP_DIR" ]] || return 0
  owner="$($STAT -c '%u' -- "$TEMP_DIR")" || return 1
  [[ "$owner" == "0" ]] || {
    printf 'ERROR: refusing to remove non-root-owned temporary directory: %s\n' "$TEMP_DIR" >&2
    return 1
  }
  "$RM" -rf --one-file-system -- "$TEMP_DIR"
  TEMP_DIR=""
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_profile_group || status=1
  if [[ -n "$LAUNCHER_PID" ]]; then wait "$LAUNCHER_PID" 2>/dev/null || true; fi
  cleanup_temp_dir || status=1
  exit "$status"
}

handle_signal() { exit "$1"; }
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

create_temp_dir() {
  local owner mode
  TEMP_DIR="$($MKTEMP -d --tmpdir=/tmp k2-ncu-profile.XXXXXXXXXX)" || die "cannot create privileged temporary directory"
  [[ "$TEMP_DIR" == /tmp/k2-ncu-profile.* && -d "$TEMP_DIR" && ! -L "$TEMP_DIR" ]] || die "mktemp returned an unsafe directory"
  owner="$($STAT -c '%u' -- "$TEMP_DIR")"
  mode="$($STAT -c '%a' -- "$TEMP_DIR")"
  [[ "$owner" == "0" && "$mode" == "700" ]] || die "temporary directory is not root-owned mode 0700"
  "$MKDIR" -m 700 -- "$TEMP_DIR/home" "$TEMP_DIR/tmp"
  TEMP_REPORT_BASE="$TEMP_DIR/k2-q6k-mmq-$RUN_ID"
  TEMP_REPORT_FILE="$TEMP_REPORT_BASE.ncu-rep"
  TEMP_TEXT_REPORT="$TEMP_DIR/k2-q6k-mmq-$RUN_ID-report.txt"
  TEMP_CSV_REPORT="$TEMP_DIR/k2-q6k-mmq-$RUN_ID-validation.csv"
  TEMP_NCU_LOG="$TEMP_DIR/k2-q6k-mmq-$RUN_ID-ncu.log"
  TEMP_REQUEST_LOG="$TEMP_DIR/k2-q6k-mmq-$RUN_ID-request.txt"
  GATE_FILE="$TEMP_DIR/start.gate"
  PID_FILE="$TEMP_DIR/profiler.pid"
}

publish_file() {
  local source="$1" destination="$2" owner mode destination_name
  [[ -f "$source" && ! -L "$source" && -s "$source" ]] || die "artifact is missing, empty, or unsafe: $source"
  reject_symlink_components "$destination"
  [[ ! -e "$destination" && ! -L "$destination" ]] || die "refusing pre-existing output path: $destination"
  [[ "$destination" == "$PROFILE_DIR/"* ]] || die "artifact destination escaped the profile directory"
  destination_name="${destination##*/}"
  [[ -n "$destination_name" && "$destination_name" != */* ]] || die "unsafe artifact destination name"

  # Walk the destination from / using directory FDs and O_NOFOLLOW, then
  # create the leaf with O_EXCL. This closes both leaf and directory-component
  # symlink races while copying from the root-only staging directory.
  "$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="/usr/bin:/bin" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
    "$PYTHON" -I -c '
import os
import stat
import sys

source, output_dir, destination_name, uid_text, gid_text = sys.argv[1:]
uid, gid = int(uid_text), int(gid_text)
if not destination_name or "/" in destination_name or destination_name in {".", ".."}:
    raise SystemExit("unsafe destination name")

directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
source_fd = destination_fd = None
created = False
try:
    for component in output_dir.strip("/").split("/"):
        next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
        os.close(directory_fd)
        directory_fd = next_fd
    directory_stat = os.fstat(directory_fd)
    if (directory_stat.st_uid, directory_stat.st_gid) != (uid, gid):
        raise SystemExit("output directory ownership changed")

    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    source_stat = os.fstat(source_fd)
    if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_uid != 0 or source_stat.st_size <= 0:
        raise SystemExit("unsafe staged artifact")

    destination_fd = os.open(
        destination_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    created = True
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            view = view[written:]
    os.fchmod(destination_fd, 0o600)
    os.fchown(destination_fd, uid, gid)
    os.fsync(destination_fd)
except BaseException:
    if destination_fd is not None:
        os.close(destination_fd)
        destination_fd = None
    if created:
        try:
            os.unlink(destination_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    raise
finally:
    if destination_fd is not None:
        os.close(destination_fd)
    if source_fd is not None:
        os.close(source_fd)
    os.close(directory_fd)

os.unlink(source)
' "$source" "$PROFILE_DIR" "$destination_name" "$TARGET_UID" "$TARGET_GID"

  [[ ! -e "$source" ]] || die "publication did not consume staged artifact: $source"
  [[ -f "$destination" && ! -L "$destination" && -s "$destination" ]] || die "published artifact is unsafe: $destination"
  owner="$($STAT -c '%u:%g' -- "$destination")"
  mode="$($STAT -c '%a' -- "$destination")"
  [[ "$owner" == "$TARGET_UID:$TARGET_GID" && "$mode" == "600" ]] || die "published artifact has incorrect ownership or mode: $destination"
}

validate_import_csv() {
  "$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="$SAFE_CHILD_PATH" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
    "$PYTHON" -I -c '
import csv
import re
import sys
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
_, name = next(iter(results))
if re.fullmatch(expected, name) is None:
    raise SystemExit(f"unexpected kernel name: {name}")
print(f"Validated one kernel result: {name}")
' "$TEMP_CSV_REPORT" "$KERNEL_NAME_RE"
}

run_check_only() {
  SCRIPT_PGID="$(read_process_field pgid "$$")" || die "cannot determine the script process group"
  is_uint "$SCRIPT_PGID" && (( SCRIPT_PGID > 1 )) || die "unsafe script process group"
  validate_destinations_absent
  printf 'CHECK OK: fixed absolute paths and executable locations\n'
  printf 'CHECK OK: four absolute Q6_K model shards resolve to pinned blob files\n'
  printf 'CHECK OK: output paths are absolute, absent, and contain no symlink components\n'
  printf 'CHECK OK: run ID, loopback binding, type-14 filter, skip 32, and count 1\n'
  printf 'CHECK OK: setsid and ps are available for gated process-group verification\n'
  printf 'CHECK ONLY: ncu and llama-server were not launched\n'
}

case $# in
  0) ;;
  1) [[ "$1" == "--check-only" ]] || die "usage: $0 [--check-only]"; CHECK_ONLY=true ;;
  *) die "usage: $0 [--check-only]" ;;
esac

if [[ ${NCU_RUN_ID+x} ]]; then RUN_ID="$NCU_RUN_ID"; else RUN_ID="run-$($DATE -u +%Y%m%dT%H%M%SZ)-$$"; fi
unset NCU_RUN_ID
validate_run_id "$RUN_ID"
validate_fixed_paths
set_output_paths

if [[ "$CHECK_ONLY" == true ]]; then run_check_only; exit 0; fi

validate_sudo_origin
validate_destinations_absent
SCRIPT_PGID="$(read_process_field pgid "$$")" || die "cannot determine the script process group"
is_uint "$SCRIPT_PGID" && (( SCRIPT_PGID > 1 )) || die "unsafe script process group"
listeners="$($SS -H -ltn "sport = :$PORT")" || die "cannot inspect TCP port $PORT"
[[ -z "$listeners" ]] || die "port $PORT is already occupied; no profiling session was started"
create_temp_dir

SERVER_COMMAND=(
  "$SERVER" --model "$MODEL_1" --host "$HOST" --port "$PORT"
  --n-gpu-layers 99 --ctx-size 8192 --parallel 1 --no-warmup --flash-attn on
  --temp 1 --top-p 1 --top-k 0 --min-p 0 --reasoning-format deepseek
  --reasoning-budget "$REASONING_BUDGET"
  --reasoning-budget-message $'\nThe reasoning budget is exhausted. Give only the requested final answer.\n'
)
NCU_COMMAND=(
  "$NCU" --target-processes application-only --replay-mode kernel --graph-profiling node
  --kernel-name-base demangled --kernel-name "$KERNEL_FILTER"
  --launch-skip "$LAUNCH_SKIP" --launch-count "$LAUNCH_COUNT"
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis
  --section Occupancy --section WarpStateStats --section LaunchStats
  --export "$TEMP_REPORT_BASE" "${SERVER_COMMAND[@]}"
)
printf 'Nsight Compute command:'; printf ' %q' "${NCU_COMMAND[@]}"; printf '\nReport destination after validation: %s\n' "$REPORT_FILE"

# The gate prevents ncu/model startup until ps verifies the setsid-created group.
"$SETSID" -- "$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="$SAFE_CHILD_PATH" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
  "$BASH_BIN" --noprofile --norc -p -c '
    set -euo pipefail
    umask 077
    readonly pid_file="$1" gate_file="$2"
    shift 2
    printf "%s\n" "$$" >"$pid_file"
    while [[ ! -e "$gate_file" ]]; do /usr/bin/sleep 0.01; done
    exec "$@"
  ' profile-ncu-gate "$PID_FILE" "$GATE_FILE" "${NCU_COMMAND[@]}" >"$TEMP_NCU_LOG" 2>&1 &
LAUNCHER_PID=$!

for ((attempt = 0; attempt < 500; attempt++)); do
  [[ -s "$PID_FILE" ]] && break
  kill -0 "$LAUNCHER_PID" 2>/dev/null || die "setsid launcher exited before process-group validation"
  "$SLEEP" 0.01
done
[[ -s "$PID_FILE" ]] || die "timed out waiting for gated profiler PID"
IFS= read -r NCU_PID <"$PID_FILE" || die "cannot read gated profiler PID"
[[ "$NCU_PID" == "$LAUNCHER_PID" ]] || die "setsid unexpectedly forked; ncu was not started"
PROFILE_PGID="$(read_process_field pgid "$NCU_PID")" || die "cannot read profiler process group"
validate_profile_group
: >"$GATE_FILE"

ready=false
for ((attempt = 0; attempt < 180; attempt++)); do
  kill -0 "$NCU_PID" 2>/dev/null || die "ncu or llama-server exited before becoming ready"
  if "$CURL" --silent --fail --max-time 1 "http://$HOST:$PORT/health" >/dev/null; then ready=true; break; fi
  "$SLEEP" 1
done
[[ "$ready" == true ]] || die "server did not become ready within 180 seconds"

printf 'Running one deterministic decode request\n'
"$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="/usr/bin:/bin" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
  K2_BASE_URL="http://$HOST:$PORT" K2_TEST_REASONING_BUDGET="$REASONING_BUDGET" K2_TEST_MAX_TOKENS="$MAX_TOKENS" \
  "$PYTHON" -I "$CLIENT" >"$TEMP_REQUEST_LOG" 2>&1

printf 'Stopping the verified profiling process group %s\n' "$PROFILE_PGID"
stop_profile_group
if wait "$LAUNCHER_PID"; then :; else
  ncu_status=$?
  LAUNCHER_PID=""; NCU_PID=""; PROFILE_PGID=""
  "$SED" -n '1,260p' "$TEMP_NCU_LOG" >&2
  die "ncu exited unsuccessfully with status $ncu_status"
fi
LAUNCHER_PID=""; NCU_PID=""; PROFILE_PGID=""

[[ -f "$TEMP_REPORT_FILE" && ! -L "$TEMP_REPORT_FILE" && -s "$TEMP_REPORT_FILE" ]] || die "ncu did not create a nonempty safe report"
"$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="$SAFE_CHILD_PATH" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
  "$NCU" --import "$TEMP_REPORT_FILE" --csv --page raw --print-kernel-base demangled >"$TEMP_CSV_REPORT"
[[ -s "$TEMP_CSV_REPORT" ]] || die "ncu CSV import was empty"
validate_import_csv
"$ENV_BIN" -i HOME="$TEMP_DIR/home" PATH="$SAFE_CHILD_PATH" LC_ALL=C TMPDIR="$TEMP_DIR/tmp" \
  "$NCU" --import "$TEMP_REPORT_FILE" --page details --print-details all --print-kernel-base demangled >"$TEMP_TEXT_REPORT"
[[ -s "$TEMP_TEXT_REPORT" ]] || die "ncu details import was empty"

validate_destinations_absent
publish_file "$TEMP_REPORT_FILE" "$REPORT_FILE"
publish_file "$TEMP_TEXT_REPORT" "$TEXT_REPORT"
publish_file "$TEMP_NCU_LOG" "$NCU_LOG"
publish_file "$TEMP_REQUEST_LOG" "$REQUEST_LOG"
printf 'Profiling complete\nReport: %s\nText:   %s\n' "$REPORT_FILE" "$TEXT_REPORT"
