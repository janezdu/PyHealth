#!/bin/bash
# Test 2 on the DROPOUT-ONLY controls -- the comparison that decides whether the
# best-of-K gain on this axis is XM or is dropout again.
#
# On Test 1 the same control inverted the conclusion: dropout 0.3 alone scored
# 0.496 / 0.923, essentially matching XM K=4's 0.570 / 0.923 with no best-of-K
# at all. If it also matches xm4's 0.0139 overall AP here, the Test 2 result is
# dropout too.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-doctrl
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" \
    --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save \
    --run do01_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.1_save \
    --run do01_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.1_xm4_save \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_dropout_ctrl_budget8000.json
