#!/bin/bash
# NATURAL-mix RLSP for the six SHARED-generator arms. Sized for a 1h slot.
#
# Cut down from _test2_rlsp_all.sh, which is ~3h. Two things were dropped:
#   - the upsample half (that cell is already scored for most of these arms)
#   - the four per-hospital arms (fedavg_ft, lat_lora_ft, ditto0.1, perada),
#     each of which needs 8 tstr + 8 rlsp classifiers per seed against a shared
#     generator's 1 + 8. Those four alone were ~64 classifiers per seed.
#
# What is left is exactly what the natural-mix figure needs: the four arms
# already on the upsample figure, plus the two latent best-of-K arms.
#
# ~62 classifiers per seed against t2-rlsp-irm4's ~45 in 23m43s (~10.6s each),
# so ~11 min per seed and ~33 min for all three. Natural mix trains on FEWER
# records than upsample (the real half is not duplicated), so that estimate is
# an over-estimate rather than an under-.
#
# Seeds loop OUTER and each seed writes one complete file, so an interrupt
# after seed 0 still leaves a usable file. Completed files are skipped on
# re-run.
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_irm=_outputs/fedavg_full_hilo8_random_irmrho100_save
      --run fed_xm4_do03=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run fedavg_irm_xm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.1_xm4_save
      --run lat_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_z16_xm4_save
      --run lat_xm4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_z16_xm4_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for seed in 0 1 2; do
    out="$RES/test2_rlspnat_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ rlsp natural, seed $seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
