#!/bin/bash
# The frontier figure on the UNWEIGHTED arms, with the REWEIGHTED centralized
# borrowed in as the pooled-data anchor -- hence the arm name centralized_rw.
#
# Source is the purpose-built merged pair mix_spec.json / mix_gen.json. Every
# arm on this figure comes from there and nowhere else. Do NOT rebuild it from
# test1_baselines + test1_fedxm4 + test1_xmirm: those files carry a genuinely
# different centralized (the UNWEIGHTED one, at -44/-53 on all codes and
# -70/-81 on rare, far off any sensible axis) and have no fedavg_ft arm at all
# -- their nearest thing is lora_head_ep2, which is a separate adapter variant
# and NOT this point.
#
#   bash examples/fedpyhealth/scripts/plot_unweighted.sh
#   bash examples/fedpyhealth/scripts/plot_unweighted.sh --metric Pearson
#   bash examples/fedpyhealth/scripts/plot_unweighted.sh --lim=-1.4,1.25
#
# The frontier line runs local -> centralized_rw: the two no-method regimes at
# opposite ends of the trade-off, which is what the shaded region is supposed
# to mean. Both are on-scale here, so the two-point line is well determined.
# Any flag is passed straight through to plot_rw_frontier.py.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
T=_outputs/results/tests
exec python examples/fedpyhealth/scripts/plot_rw_frontier.py \
    --spec $T/mix_spec.json \
    --gen  $T/mix_gen.json \
    --centralized-label centralized \
    --frontier local,centralized_rw \
    --out _outputs/results/unweighted_frontier.png "$@"
