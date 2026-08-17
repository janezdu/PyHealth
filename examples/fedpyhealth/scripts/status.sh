#!/bin/bash
# Are my jobs healthy? One command, safe to run on the login node.
#
# Run from the repo root:
#   bash examples/fedpyhealth/scripts/status.sh          # queue + progress + health
#   bash examples/fedpyhealth/scripts/status.sh --sweep  # also sweep finished runs
#
# Prints, in order:
#   - every queued/running job with its phase, round progress and ETA
#   - trouble found inside a RUNNING job (OOM, traceback, NaN loss) -- a job can
#     report RUNNING for hours after it has stopped making progress
#   - pending jobs that can never start (dead dependency), with the scancel line
#   - how recently-ended jobs ended, and which results have landed on disk
#
# Reads squeue and the log files only. It never starts work and never touches
# the GPU, so it is fine to run repeatedly while an experiment is in flight.
set -euo pipefail

cd "$(dirname "$0")/../../.."          # repo root; every path below is relative

PY=.venv/bin/python                     # this clone runs from .venv, not pixi
[ -x "$PY" ] || PY=python

"$PY" examples/fedpyhealth/main.py status

if [ "${1:-}" = "--sweep" ]; then
    echo
    echo "=== sweeping newly-finished runs ==="
    "$PY" examples/fedpyhealth/exp_log.py check --sort Prevalence_R2
fi
