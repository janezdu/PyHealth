#!/bin/bash
# RLSP -- real local + synthetic pooled. One classifier per hospital trained on
# that site's own real records PLUS the regime's pooled synthetic set.
#
# WHY. Every synthetic arm scored so far asks whether synthetic REPLACES real,
# and the answer under logreg is no: the best of them (xm4, 0.604 AUROC) loses
# to a hospital's own 142-1599 real records (real_local, 0.644). Nobody deploys
# that way. RLSP asks the question a site would actually ask -- I keep my data
# and ALSO take what the federation generated, am I better off?
#
# THE COMPARISON IS PER SITE. rlsp:<regime>:<hid> against real_local:<hid>, not
# against the pooled arms. Below its own real_local means the synthetic data is
# actively hurting that site, and the smallest sites (429 at 142 records, 358
# at 179) are where it has the most room to help.
#
# THE RNN VERSION, and it asks a sharper question than the logreg one.
#
# The RNN scores AUROC 0.417 on real_local -- below chance -- and that failure
# tracks training size at r = +0.84 across the eight sites: 0.359 at 142
# records, 0.550 at 1599. It is a sample-size collapse.
#
# RLSP hands each of those starved classifiers ~6,600 extra records. If the RNN
# recovers here, synthetic data is doing exactly what a small site needs it for
# -- supplying volume -- even though it loses to real data head-to-head. That
# would be a use for these generators that every arm scored so far has missed,
# because every arm so far asked synthetic to REPLACE real rather than pad it.
#
# Read it three ways: against rnn real_local (did the collapse lift), against
# the logreg RLSP run (does the classifier still matter once data is plentiful),
# and against the pooled synthetic arms (is the real half contributing at all).
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-rlsp-rnn
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
# GPU stays visible: the RNN wants it.
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

# Three regimes only: the best centralized arm, its federated twin, and plain
# fedavg. Each adds 8 classifiers, so the arm set grows fast.
RUNS="--run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run fed_xm4_do03=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save"

for seed in 0 1 2; do
    out="$RES/test2_head_rlsp_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ rlsp, seed $seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
