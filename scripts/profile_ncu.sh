#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  'ERROR: this full-model Nsight Compute workflow is permanently disabled.' \
  '' \
  'It loaded the approximately 60 GB K2 model, exhausted GPU graphics-context' \
  'memory, and froze the GNOME desktop. It must not be run again.' \
  '' \
  'Historical implementation: scripts/profile_ncu.full-model.disabled.sh' \
  'Safe replacement: docs/safe-ncu-microbenchmark.md' >&2
exit 64

