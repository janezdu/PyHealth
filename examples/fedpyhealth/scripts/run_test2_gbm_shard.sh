#!/bin/bash
# Table 2's GBM column, one shard at a time, to fit cpu-interactive's 1h cap.
#
# WHY SHARD AT ALL. A whole GBM seed is ~2h (133 classifier fits at ~54s each),
# against a 1h MaxTime on cpu-interactive and a QoS of one job at a time. The
# regular `cpu` partition would run it unsharded but ranks this account 5th
# from the bottom of ~1,600 pending jobs.
#
# WHY SHARDING IS SAFE. The scored code set is chosen from the manifest and the
# real evaluation fold alone -- test2_rare_efficacy.py:975-1034 -- and never
# from --run. Verified empirically: eval_codes is identical between the logreg
# and gbm files at seed 0 and differs across seeds. So every shard scores the
# SAME 30 codes and the per-arm numbers are directly comparable.
#
# WHAT EACH SHARD COSTS. Per-hospital regimes are 16 classifiers (8 tstr +
# 8 rlsp); fedavg and fedprox share one global generator, so they are 9. TSTR
# cannot be skipped -- RLSP pools its synthetic from those arms.
#
#   shard 1  fedavg, fedprox, xm4 + real_local + pooled   44  ~40 min
#   shard 2  fedavg_ft, lat_lora_ft                       32  ~29 min
#   shard 3  ditto0.1, perada                             32  ~29 min
#   shard 4  fedrep, lgfedavg                             32  ~29 min
#
# Only shard 1 computes real_local and the pooled reference arms; the rest pass
# --skip-real-local so they are not recomputed four times.
#
#   export FEDCOHORT_CACHE=/path/to/cohort/caches
#   SEED=1 SHARD=1 bash examples/fedpyhealth/scripts/run_test2_gbm_shard.sh
#
# Then merge:
#   python examples/fedpyhealth/scripts/merge_test2_shards.py --seed 1
#
# TRY MORE CORES ON SHARD 1. The unsharded run kept only 4.67 of 8 cores busy,
# but HistGradientBoostingClassifier threads its histogram build over OMP, so
# --cpus-per-task=32 may cut the wall time enough to fold two shards into one
# hour. The script prints its own elapsed time so shard 1 calibrates the rest.
set -euo pipefail
: "${SEED:?set SEED to the eval seed (1 or 2)}"
: "${SHARD:?set SHARD to 1, 2, 3 or 4}"
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
[ -d "$CACHE" ] || { echo "no cohort cache at $CACHE" >&2; exit 1; }
RES=_outputs/results/tests
mkdir -p "$RES"
export CUDA_VISIBLE_DEVICES=
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python

D=_outputs
A_fedavg="--run fedavg=$D/fedavg_full_hilo8_random_nes_save"
A_fedprox="--run fedprox=$D/fedprox_E2_R50_hilo8_random_privacy_prox0.01_nes_save"
A_xm4="--run xm4=$D/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save"
A_fedavg_ft="--run fedavg_ft=$D/fedavg_ft_full_hilo8_random_nes_save"
A_lat="--run lat_lora_ft=$D/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save"
A_ditto="--run ditto0.1=$D/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save"
A_perada="--run perada=$D/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save"
A_fedrep="--run fedrep=$D/fedrep_E2_R50_hilo8_random_privacy_nes_save"
A_lgfedavg="--run lgfedavg=$D/lg_fedavg_E2_R50_hilo8_random_privacy_nes_save"

case "$SHARD" in
  1) RUNS="$A_fedavg $A_fedprox $A_xm4"; EXTRA="" ;;
  2) RUNS="$A_fedavg_ft $A_lat";         EXTRA="--skip-real-local" ;;
  3) RUNS="$A_ditto $A_perada";          EXTRA="--skip-real-local" ;;
  4) RUNS="$A_fedrep $A_lgfedavg";       EXTRA="--skip-real-local" ;;
  *) echo "SHARD must be 1, 2, 3 or 4" >&2; exit 1 ;;
esac

for d in $(echo "$RUNS" | grep -o "$D/[^ ]*_save"); do
  [ -f "$d/synthetic.json" ] || { echo "MISSING $d/synthetic.json" >&2; exit 1; }
done

OUT="$RES/test2_clf_gbm_seed${SEED}_shard${SHARD}.json"
[ -f "$OUT" ] && { echo "already done: $OUT"; exit 0; }

echo "### gbm seed $SEED shard $SHARD  ($(date +%H:%M:%S))"
START=$(date +%s)
$PY examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS $EXTRA \
    --rlsp --classifier gbm \
    --fold test --code-pool all --min-positives 20 \
    --n-eval-codes 30 --eval-seed "$SEED" \
    --train-budget 8000 --resume --out "$OUT"
echo "### shard $SHARD done in $((($(date +%s)-START)/60)) min  ($(date +%H:%M:%S))"
echo "Use that number to decide whether the next shard needs its own hour."
