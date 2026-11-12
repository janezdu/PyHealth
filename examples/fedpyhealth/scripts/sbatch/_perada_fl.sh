#!/bin/bash
# PerAda as the paper specifies it (--regime perada_fl), replacing the post-hoc
# `perada` arm, which ran KD client-side toward a frozen global.
#
# BUDGET, 100 passes/hospital like every arm on the frontier:
#   trunk   20 FedAvg rounds x 2 local epochs, full model      =  40
#   PerAda  30 rounds x (1 ep local adapter + 1 ep personal)   =  60
# --ft-epochs 60 is what sets that split: the equal-budget code already gives
# FT regimes (total - ft_epochs) // local_epochs = 20 trunk rounds, and
# run_perada takes ft_epochs / 2 = 30 rounds.
#
# Adapter lora_head r8 (as FedLIBRA). lambda 0.1 (as the Ditto row). Server KD:
# 10 steps/round on a 1,000-record pool sampled once from the frozen trunk.
# Everything else matches the other baselines: E2, no early stopping, sample
# weighting, privacy metrics.
#
# Train + generate + Test 1 in one task. ~25 min train by analogy with Ditto
# (50 FedAvg rounds took 15-25 min; PerAda rounds only backprop through the
# head), ~13 min generation; 2h is margin.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=perada-fl
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
NAME=perada_fl

ARGS="--profile full --cohort-cache $CACHE --local-epochs 2 --n-rounds 50 \
      --metrics privacy --no-early-stop --regime perada_fl --ft-epochs 60 \
      --adapter lora_head --adapter-rank 8"

echo "### PerAda (faithful)  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py $ARGS --resume

D=$(python examples/fedpyhealth/train.py $ARGS --print-save-dir | tail -1)
echo "### save_dir = $D"

echo "### generating at 8000/hospital"
python examples/fedpyhealth/generate.py --save-dir "$D" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="_outputs/results/tests/test1_${NAME}_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" \
        --run "$NAME=$D" --fold test --synth-cap 8000 --rare-scope pooled \
        --real-scope "$scope" --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done
echo done
