#!/bin/bash
# The two scoring passes still sitting in Delta's queue, as a plain shell script
# for a workstation. No SLURM, no GPU -- both are CPU-only, and Delta's CPU
# fairshare (0.0197) puts them near the bottom of a 1,600-job queue, so a PC
# that is idle beats a cluster that will not schedule them.
#
# WHAT IT NEEDS. The cohort cache (any machine that has trained already has it;
# never rebuild it just for this) and the synthetic.json of each run being
# scored. Nothing else -- no checkpoints, no GPU, no training.
#
#   export FEDCOHORT_CACHE=$HOME/cache/fedcohort     # holds hilo8_random/
#   bash examples/fedpyhealth/scripts/run_cpu_scoring_local.sh fig1
#   bash examples/fedpyhealth/scripts/run_cpu_scoring_local.sh gbm
#   bash examples/fedpyhealth/scripts/run_cpu_scoring_local.sh both
#
# Both passes skip any output file already present, so re-running is safe and
# an interrupted pass resumes at the next missing file.
set -euo pipefail
WHICH="${1:-both}"
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
[ -d "$CACHE" ] || { echo "no cohort cache at $CACHE" >&2; exit 1; }
RES=_outputs/results/tests
mkdir -p "$RES"
# scoring never touches a model; make that impossible to get wrong
export CUDA_VISIBLE_DEVICES=
PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python

# ---------------------------------------------------------------- Figure 1 CIs
# 6 arms x 3 independent runs at batch 128, published per-arm treatment.
# Minutes, not hours. Aggregate as the main table does: mean over the 8
# hospitals of PrevVal_All_Prevalence_R2, then mean and SD over the three runs.
fig1 () {
  local RUNS="" rep arm d miss=0
  for rep in 2 3 4; do
    for arm in local centralized fedavg ft_full ft_lora fedlibra; do
      d="_outputs/b128rep${rep}_${arm}_save"
      if [ -f "$d/synthetic.json" ]; then RUNS="$RUNS --run ${arm}_rep${rep}=$d"
      else echo "MISSING $d/synthetic.json" >&2; miss=1; fi
    done
  done
  [ "$miss" = 0 ] || { echo "copy the 18 synthetic.json files over first" >&2; exit 1; }
  for scope in hospital pooled; do
    local name=spec; [ "$scope" = "pooled" ] && name=gen
    local out="$RES/test1_b128reps_${name}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "### Figure 1, $name  ($(date +%H:%M:%S))"
    $PY examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
  done
}

# ------------------------------------------------- Table 2's GBM column, seeds 1-2
# seed 0 is already scored on Delta. 10 arms x 8 hospitals x 30 codes of
# HistGradientBoostingClassifier per seed -- this is the slow half, and the
# half a fast desktop CPU helps most.
gbm () {
  local RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_nes_save
      --run lat_lora_ft=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save
      --run ditto0.1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save
      --run perada=_outputs/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save
      --run fedprox=_outputs/fedprox_E2_R50_hilo8_random_privacy_prox0.01_nes_save
      --run fedrep=_outputs/fedrep_E2_R50_hilo8_random_privacy_nes_save
      --run lgfedavg=_outputs/lg_fedavg_E2_R50_hilo8_random_privacy_nes_save
      --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save"
  local d miss=0
  for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "MISSING $d/synthetic.json" >&2; miss=1; }
  done
  [ "$miss" = 0 ] || { echo "copy the 9 synthetic.json files over first" >&2; exit 1; }
  for seed in ${SEEDS:-1 2}; do
    local out="$RES/test2_clf_gbm_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "### Table 2 gbm, eval-seed $seed  ($(date +%H:%M:%S))"
    $PY examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --classifier gbm \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
  done
}

case "$WHICH" in
  fig1) fig1 ;;
  gbm)  gbm ;;
  both) fig1; gbm ;;
  *) echo "usage: $0 [fig1|gbm|both]" >&2; exit 1 ;;
esac

echo
echo "done ($(date +%H:%M:%S))"
echo "Copy back only _outputs/results/tests/*.json -- a few hundred KB each."
