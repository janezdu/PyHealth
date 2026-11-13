#!/bin/bash
# Table 2 under a second and third downstream classifier.
#
# WHY. Table 2's utility column is one RNN, and that RNN overfits: trained on
# a single hospital's 142-1,832 real records it scores AUROC 0.420, below
# chance, which all three reviewers flagged. The existing logistic-regression
# control lifts that same arm to 0.644 AND REVERSES THE ARM ORDERING -- under
# the RNN, FedAvg beats a site's own data by +0.198; under logreg it loses by
# 0.046.
#
# So "which method is best" currently depends on the downstream model. A third
# point on the capacity axis says whether the RNN is the outlier or whether
# the two disagree for a deeper reason. Reporting all three is also a stronger
# answer than quietly switching to whichever one flatters the method.
#
# THE THREE DIFFER ONLY IN MODEL CLASS. gbm and logreg are fit on the same
# bag_of_codes matrix, so nothing here confounds the model with its features.
# sklearn's HistGradientBoostingClassifier rather than PyHealth's XGBoostModel
# for exactly that reason: XGBoost would bring its own feature matrix (visit
# counts via nested_multihot) and a new dependency.
#
# ARMS ARE TABLE 2's OWN ROWS, each keeping its published composition --
# real_local and real_pooled are real-only, every method row is RLSP (own real
# + pooled synthetic). The natural mixture, no --rlsp-weight.
#
# logreg runs at its DEFAULT C=1.0. The existing C sweep (0.1/1.0/10) stays in
# the appendix; picking the C that most favours a conclusion is the thing to
# avoid here.
#
# CPU ONLY -- no GPU, no generation, every synthetic.json is already on disk.
# The RNN column needs no rerun: it is the published table.
#
# 10 arms x 8 hospitals x 30 codes x 3 draws, twice. logreg is minutes; gbm
# fits 30 boosted ensembles per classifier and is the slow half.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=t2-clf-columns
#SBATCH --time=06:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_nes_save
      --run lat_lora_ft=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save
      --run ditto0.1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save
      --run perada=_outputs/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save
      --run fedprox=_outputs/fedprox_E2_R50_hilo8_random_privacy_prox0.01_nes_save
      --run fedrep=_outputs/fedrep_E2_R50_hilo8_random_privacy_nes_save
      --run lgfedavg=_outputs/lg_fedavg_E2_R50_hilo8_random_privacy_nes_save
      --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

mkdir -p "$RES"
for clf in ${CLFS:-logreg gbm}; do
  for seed in ${SEEDS:-0 1 2}; do
    out="$RES/test2_clf_${clf}_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo
    echo "############ $clf, eval-seed $seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --classifier "$clf" \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --resume --out "$out"
  done
done

echo
echo "done ($(date +%H:%M:%S))"
echo "Compare against the published RNN column in"
echo "test2_rlspall_natural_rnn_seed{0,1,2}.json -- same arms, same draws."
echo "Read real_local FIRST in each: if it clears prior under a classifier,"
echo "that classifier is not the one that produced the 0.420 cell."
