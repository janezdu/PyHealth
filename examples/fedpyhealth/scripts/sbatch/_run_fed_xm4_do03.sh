#!/bin/bash
# fedavg + dropout 0.3 + best-of-K K=4 -- the federated twin of the centralized
# `xm4` arm, then generate and score it on all three Test 2 pools.
#
# WHY. Every best-of-K result so far is centralized. `xm4` (centralized, do 0.3,
# K=4) beats its matched control `do03_only` (centralized, do 0.3, K=1) on both
# Test 2 pools, but the federated K=4 arms that exist run at dropout 0.1, so
# "XM helps" has never been tested under federation at a matched dropout. This
# supplies the missing cell.
#
# THE FEDERATED K LADDER this completes, all fedavg:
#   K=4, dropout 0.3  this run                    (new)
#   K=8, dropout 0.3  xm8_fedavg                  (trained, never head-scored)
#   K=4, dropout 0.1  do01_fed_k4                 (trained, NEVER Test-2-scored)
# All are scored together below, one file per pool, so the comparison does not
# cross the ~0.002 real_pooled drift between separate scoring runs.
#
# WHAT IT STILL CANNOT SETTLE. There is no fedavg K=1 control at dropout 0.3
# either, so this arm has nothing to be ablated AGAINST under federation -- it
# tells you whether the centralized ranking transfers, not whether K causes it.
# The control is one more run with --xm-k 1; ask for it before reading a
# causal claim into this number.
#
# READ THE FIRST EPOCH. Every epoch prints `xm K=4  spread=...  win_entropy=...`
# and a spread of 0.000 means the K candidates are bit-identical and the run is
# training on 1/K of its gradient. That is the only pre-flight check that
# matters.
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=fed-xm4-do03
#SBATCH --time=06:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

ARGS="--profile full --regime fedavg --cohort-cache $CACHE --no-early-stop
      --dropout 0.3 --xm-k 4"

# Ask train.py for the directory rather than rebuilding the naming rule here --
# the component order has bitten this project twice, and a guessed name means
# the scoring step silently skips the arm.
DIR=$(python examples/fedpyhealth/train.py $ARGS --print-save-dir)
echo "save_dir = $DIR"

python examples/fedpyhealth/train.py $ARGS
python examples/fedpyhealth/generate.py --save-dir "$DIR" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000

# Test 1, CPU-cheap, both scopes.
for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" \
        --run fed_xm4_do03="$DIR" \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "$RES/test1_fedxm4_${tag}_b200.json"
done

# Test 2 on the tail, alongside the centralized pair it is meant to mirror.
python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" \
    --run fed_xm4_do03="$DIR" \
    --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save \
    --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save \
    --run xm8_fedavg=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save \
    --run do01_fed_k4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.1_xm4_save \
    --fold test --train-budget 8000 \
    --out "$RES/test2_fedxm4_budget8000.json"

# Test 2 on the head-code pool, three draws, same settings as the existing run
# so the numbers drop straight into the visualisation's head panel.
for seed in 0 1 2; do
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" \
        --run fed_xm4_do03="$DIR" \
        --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save \
        --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save \
        --run xm8_fedavg=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save \
        --run do01_fed_k4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.1_xm4_save \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 \
        --out "$RES/test2_head_fedxm4_seed${seed}.json"
done
echo "done"
