#!/bin/bash
# Score the two federated best-of-K arms that were trained but never evaluated
# on the pools where this test can actually separate arms.
#
#   xm8_fedavg    fedavg, dropout 0.3, K=8  -- has Test 1 + tail Test 2, no head
#   do01_fed_k4   fedavg, dropout 0.1, K=4  -- has Test 1, NO Test 2 at all
#
# Both centralized anchors (xm4, do03_only) are rescored in the SAME file so the
# federated arms are compared against them under one classifier draw, rather
# than across files, where real_pooled drifts ~0.002 between runs.
#
# No training and no generation -- every synthetic.json is already on disk, so
# this is CPU work and takes about ten minutes.
#
#   srun --account=bgyw-delta-gpu --partition=gpuA40x4 --gpus-per-node=1 \
#        --cpus-per-task=8 --mem=32g --time=01:00:00 --pty bash
#   source .venv/bin/activate
#   bash examples/fedpyhealth/scripts/score_fed_xm_cpu.sh
#
# WHAT THIS STILL CANNOT SETTLE. No fedavg arm has a K=1 control at any dropout,
# so this gives the federated K ladder but no ablation baseline under it. A gap
# here is not evidence that K does nothing federated.
set -euo pipefail

CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"
RES=_outputs/results/tests
if [ -z "${CUDA_VISIBLE_DEVICES+set}" ]; then export CUDA_VISIBLE_DEVICES=; fi

RUNS="--run xm8_fedavg=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save
      --run do01_fed_k4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.1_xm4_save
      --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save"

for seed in 0 1 2; do
    out="$RES/test2_head_fedladder_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ head-code draw, seed $seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done

out="$RES/test2_fedladder_budget8000.json"
if [ -f "$out" ]; then echo "skip $out"; else
    echo "############ tail pool, for continuity with the older numbers"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --fold test --train-budget 8000 --out "$out"
fi
echo
echo "Read xm8_fedavg and do01_fed_k4 against xm4 in the SAME file. The"
echo "centralized/federated split is a confound here, not a result."
