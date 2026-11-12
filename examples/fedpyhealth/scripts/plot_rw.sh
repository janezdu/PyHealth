#!/bin/bash
# Redraw the reweighted frontier figure. Plain CPU, about ten seconds --
# matplotlib over two already-scored JSONs, no model, no GPU, no scheduler.
#
#   bash examples/fedpyhealth/scripts/plot_rw.sh
#   bash examples/fedpyhealth/scripts/plot_rw.sh --metric Pearson
#   bash examples/fedpyhealth/scripts/plot_rw.sh --lim -0.6,1.1
#
# Any flag is passed straight through to plot_rw_frontier.py; --out defaults to
# _outputs/results/rw_frontier.png.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
exec python examples/fedpyhealth/scripts/plot_rw_frontier.py "$@"
