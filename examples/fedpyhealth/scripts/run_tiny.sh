#!/bin/bash
# Tiny run: the whole pipeline on the REAL frozen cohort, at small scale.
#
# Same data and same 8 clients as a full run -- only the compute is small
# (1 FedAvg round, 1 local epoch, 64 synthetic patients). It answers "does this
# run end to end and produce non-degenerate output", never "is this any good":
# at num_synth=64 prevalence resolves only to 1/64, so Test 1's R^2 here is
# quantization noise.
#
# Run from the repo root:
#   bash examples/fedpyhealth/scripts/run_tiny.sh
#   bash examples/fedpyhealth/scripts/run_tiny.sh --regime local
#   bash examples/fedpyhealth/scripts/run_tiny.sh --dry-run
#
# This does not itself run on the login node -- main.py pre-flights the cohort
# (a second of JSON reading), then submits the GPU job and prints its id.
set -euo pipefail

python examples/fedpyhealth/main.py train --profile tiny "$@"
