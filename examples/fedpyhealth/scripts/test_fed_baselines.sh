#!/bin/bash
# Unit tests for the FL-baseline mechanisms. CPU, seconds.
#   bash examples/fedpyhealth/scripts/test_fed_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python -m pytest tests/core/test_fedparts.py tests/core/test_fedreg.py -q "$@"
