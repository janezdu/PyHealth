#!/bin/bash
# Score the IRM sweep at n=200 alongside the arms already on the plot.
# CPU work -- run it on an interactive node you already hold, not as a GPU job:
#
#   source .venv/bin/activate
#   bash examples/fedpyhealth/scripts/score_irm_sweep.sh
#
# The reference arm (rho=1e2, warmup=17, step) is included so the sweep is read
# against it rather than against a number from a different scoring run.
set -euo pipefail
CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"
RES=_outputs/results/tests
# Two naming schemes coexist here and it matters. The REFERENCE arm was
# launched through main.py, which builds its own run name (fedavg_full_...).
# The sweep calls train.py directly, so it gets train.py's native name
# (fedavg_E2_R50_...). They are the same program; only the launcher differs.
REF=_outputs/fedavg_full_hilo8_random_irmrho100_save
B=_outputs/fedavg_E2_R50_hilo8_random_privacy

RUNS="--run ref_rho100_w17=$REF"
add () { [ -d "$2" ] && RUNS="$RUNS --run $1=$2" || echo "skip (missing) $1: $2" >&2; }
add rho1        "${B}_irm1w17_save"
add rho10       "${B}_irm10w17_save"
add warm0       "${B}_irm100_save"
add warm34      "${B}_irm100w34_save"
add geom        "${B}_irm100w17_geom_save"
add cosine      "${B}_irm100w17_cosine_save"
add decay       "${B}_irm100w17_decay_save"

echo "arms:$RUNS"; echo
for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "$RES/test1_irmsweep_${tag}_b200.json"
done
echo
echo "NOTE: the run-name guesses above must be checked against what train.py"
echo "actually produced -- ask it rather than guessing:"
echo "  python examples/fedpyhealth/train.py --profile full --regime fedavg \\"
echo "      --cohort-cache \$FEDCOHORT_CACHE/hilo8_random \\"
echo "      --irm-rho 100 --irm-warmup 17 --irm-schedule geom --print-save-dir"
