#!/bin/bash
# Test 2 on WELL-SUPPORTED codes instead of the tail. CPU-viable.
#
# WHY. On hilo8_random the rare pool is 548 of 921 codes and half of them carry
# 1-4 test positives, so the macro AP sits pinned at the base rate (prior 0.0067)
# and macro AUROC is estimated from a handful of points. The control proves the
# task, not the generators, is the problem: real_local -- a hospital's OWN REAL
# records -- scores AUROC 0.401, below the 0.5 floor and below every synthetic
# arm. A test whose real-data control cannot beat chance cannot rank generators.
#
# WHY NOT --code-pool common. The cohort calls a code rare at prevalence <= 5%,
# which in a 921-code ICD vocabulary is almost everything: the complement is 55
# codes and only 4 of them clear 5 test positives. "Non-rare" is not a usable
# pool here. Selecting on SUPPORT is, and that is what --min-positives does --
# the rejection step of a rejection sample, with sample_eval_codes drawing
# uniformly from what survives. The draw stays random so the head is not
# cherry-picked; only the unmeasurable codes are rejected.
#
#   --min-positives 20 -> 91 codes survive, a 30-draw has median support 55
#   --min-positives 50 -> 46 codes survive, a 30-draw has median support 97
#
# READ IT AGAINST THE FLOOR, WHICH MOVES. AP's floor is the base rate, so a
# higher-prevalence pool has a higher floor: ~0.03 here against 0.0067 on the
# tail. Compare arms to the `prior` row in the same file, never across pools.
#
# CPU. --n-eval-codes sets mask-folds to 1, so this is 13 classifiers instead of
# 52. There is no CUDA reference in test2_rare_efficacy.py; PyHealth's Trainer
# takes whatever torch reports, and CUDA_VISIBLE_DEVICES= forces CPU.
#
#   srun --account=bgyw-delta-gpu --partition=gpuA40x4 --gpus-per-node=1 \
#        --cpus-per-task=8 --mem=32g --time=01:00:00 --pty bash
#   source .venv/bin/activate
#   bash examples/fedpyhealth/scripts/test2_headcodes_cpu.sh
#
# Runs three seeds. A 30-code macro is an estimate, so the spread across seeds
# IS the error bar -- report it, do not average the three into one number.
set -euo pipefail

CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"
RES=_outputs/results/tests
MINPOS="${MINPOS:-20}"
NCODES="${NCODES:-30}"
SEEDS="${SEEDS:-0 1 2}"
if [ -z "${CUDA_VISIBLE_DEVICES+set}" ]; then export CUDA_VISIBLE_DEVICES=; fi

RUNS="--run centralized=_outputs/centralized_full_hilo8_random_nes_save
      --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_irm100=_outputs/fedavg_full_hilo8_random_irmrho100_save
      --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save
      --run lat_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_z16_xm4_save"

for seed in $SEEDS; do
    out="$RES/test2_head_min${MINPOS}_n${NCODES}_seed${seed}.json"
    if [ -f "$out" ]; then echo "skip $out"; continue; fi
    echo "############ head-code draw, seed $seed"
    python examples/fedpyhealth/test2_rare_efficacy.py \
        --cohort-cache "$CACHE" $RUNS \
        --fold test --code-pool all --min-positives "$MINPOS" \
        --n-eval-codes "$NCODES" --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo
echo "Compare each tstr: row to the prior and real_local rows IN THE SAME FILE."
echo "If real_local still cannot beat prior, the task is still the problem."
