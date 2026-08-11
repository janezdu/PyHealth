#!/bin/bash
# Full run: production scale on the frozen cohort. This is the real experiment.
#
# Run from the repo root:
#   bash examples/fedpyhealth/scripts/run_full.sh                  # fedavg
#   bash examples/fedpyhealth/scripts/run_full.sh --regime local
#   bash examples/fedpyhealth/scripts/run_full.sh --n-rounds 100
#
# For the whole four-regime table in one go, use main.py directly:
#   python examples/fedpyhealth/main.py all --profile full
set -euo pipefail

python examples/fedpyhealth/main.py train --profile full "$@"
