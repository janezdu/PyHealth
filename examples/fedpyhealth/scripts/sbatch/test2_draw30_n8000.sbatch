#!/bin/bash
# Test 2 in DRAW MODE: 30 randomly-chosen rare codes, three seeds.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/test2_draw30_n8000.sbatch
#
# The default K=4 run partitions all 478 scoreable codes into four folds and
# trains four classifiers per arm, stripping one fold each time -- a census, but
# it removes a quarter of the rare tail from every input. Draw mode strips only
# the 30 drawn codes, so 94% of the rare-to-rare co-occurrence survives instead
# of 75%, which for a 0.3%-prevalence code is often the only signal there is.
#
# The cost is statistical: the macro mean is an estimate over 30 codes, not a
# census, so this runs three seeds. Each seed draws a DIFFERENT code set, so the
# three are repeat measurements of the same quantity -- report their spread, and
# never quote one seed alone.
#
# Same synthetic data and same matched budget as the K=4 run it sits beside, so
# the two are directly comparable on everything except the masking scheme.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-draw30
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_save \
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_save \
      --run centralized=_outputs/centralized_full_hilo8_random_save \
      --run local=_outputs/local_full_hilo8_random_save"

for SEED in 0 1 2; do
  echo "############ draw-30, seed $SEED"
  python examples/fedpyhealth/test2_rare_efficacy.py \
      --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" $RUNS \
      --fold test --train-budget 8000 \
      --n-eval-codes 30 --eval-seed $SEED \
      --out "_outputs/results/tests/test2_draw30_n8000_seed${SEED}.json"
done
