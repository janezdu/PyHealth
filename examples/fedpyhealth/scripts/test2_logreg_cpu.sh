#!/bin/bash
# Test 2 with an L2 logistic regression instead of the RNN, on the head-code
# pool, same three draws as the RNN run -- so every number here has a matched
# RNN counterpart in test2_head_min20_n30_seed*.json.
#
# THE ROW THAT MATTERS IS real_local. Under the RNN it scores AUROC 0.40 on
# real data: below chance, below prior, below every synthetic arm. A real-data
# control that cannot beat guessing means the classifier is the problem. If
# real_local clears prior here, a chunk of what Test 2 has reported all along
# was RNN overfitting rather than signal, and the arm rankings need re-reading.
#
# CPU, no GPU, no generation -- every synthetic.json is already on disk.
# About fifteen minutes.
#
#   srun --account=bgyw-delta-gpu --partition=gpuA40x4 --gpus-per-node=1 \
#        --cpus-per-task=8 --mem=32g --time=01:00:00 --pty bash
#   source .venv/bin/activate
#   bash examples/fedpyhealth/scripts/test2_logreg_cpu.sh
set -euo pipefail

CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"
RES=_outputs/results/tests
if [ -z "${CUDA_VISIBLE_DEVICES+set}" ]; then export CUDA_VISIBLE_DEVICES=; fi
CS="${LOGREG_C:-0.1 1.0 10}"

RUNS="--run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save
      --run fed_xm4_do03=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run xm8_fedavg=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save
      --run lat_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_z16_xm4_save
      --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save"

# Sweeping C tests the overfitting reading directly: stronger regularisation
# (C=0.1) should help the 142-record sites most, so if real_local only clears
# prior at low C, the effect is pinned to sample size rather than to model class.
for C in $CS; do
  for seed in 0 1 2; do
    out="$RES/test2_head_logreg_c${C}_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ logreg, C=$C, seed $seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --classifier logreg --logreg-c "$C" \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
  done
done
echo
echo "Compare real_local and prior FIRST, then the arm ordering against"
echo "test2_head_min20_n30_seed*.json (same draws, RNN classifier)."
