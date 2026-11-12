#!/bin/bash
# RLSP for the FOUR-ARM fedavg comparison: vanilla / IRM / best-of-K / both.
#
# WHY THIS EXISTS. Every RLSP run so far scored the same three arms -- fedavg,
# fed_xm4_do03, xm4 -- so the "does synthetic data help a site that already has
# real records?" figure could never show IRM at all. IRM was Test-2 scored, but
# only ever as TSTR (synthetic REPLACES real), which is not the question. This
# adds the two missing arms so all four fedavg variants can be read on the same
# axis:
#
#   fedavg          plain FedAvg
#   fedavg_irm      + IRM rho=1e2, warmup 17
#   fed_xm4_do03    + best-of-K K=4
#   fedavg_irm_xm   + IRM AND best-of-K
#
# xm4 (centralized best-of-K) is kept because it costs nothing here and is the
# pooled-data reference the four federated arms are trying to reach.
#
# ---------------------------------------------------------------------------
# THE CONFOUND, stated up front because it is NOT removable from existing runs.
# The four arms do not share dropout or batch size:
#
#   arm             irm_rho  warmup  xm_k  dropout  batch
#   fedavg              -       -      -     0       128
#   fedavg_irm        100      17      -     0       128
#   fed_xm4_do03        -       -      4    0.3      256
#   fedavg_irm_xm     100      17      4    0.1      256
#
# Dropout is intrinsic to best-of-K (at dropout 0 the K passes are identical,
# so it is best-of-1 at K x the cost), but 0.3 vs 0.1 across the two xm arms is
# not intrinsic -- it is what was trained. A clean do-0.3 IRM+best-of-K run does
# NOT exist; the only do-0.3 IRM+xm dirs also carry self-training (st0.5r20@1),
# which would add a third difference. So do-0.1 is the least-confounded choice
# available, and the xm-vs-xm gap still mixes IRM with a dropout change.
#
# This confound is already present in the published three-arm RLSP figure
# (fedavg is do 0 / batch 128 against fed_xm4_do03's do 0.3 / batch 256). This
# job does not make it worse, but it does not fix it either. Fixing it needs a
# retrain, not a re-score.
# ---------------------------------------------------------------------------
#
# Settings are copied EXACTLY from _test2_rlsp_w50_upsample_rnn.sh so the new
# arms land on the same axis as the scored ones: RNN, 50/50 upsample, 30 drawn
# codes, min 20 positives, budget 8000, seeds 0/1/2. Do not "improve" them here
# or the figure stops being comparable.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-irm4
#SBATCH --time=06:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
# GPU stays visible: the RNN wants it.
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

# Five regimes x 8 hospitals = 40 RLSP classifiers per seed, plus the 8
# real_local controls. That is why --time is 6h rather than the 3h the
# three-arm version used.
RUNS="--run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run fed_xm4_do03=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_irm=_outputs/fedavg_full_hilo8_random_irmrho100_save
      --run fedavg_irm_xm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.1_xm4_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for seed in 0 1 2; do
    out="$RES/test2_rlsp_irm4arm_upsample_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ rlsp irm 4-arm, seed $seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --rlsp-weight 0.5 --rlsp-weight-mode upsample --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
