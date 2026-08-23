#!/bin/bash
# Re-score BOTH tests against the current 8,000-per-hospital synthetic data,
# with the two controls that make the arms comparable.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/rescore_n8000.sbatch
#
# Why both, and why controlled:
#
#  test1 --synth-cap 8000
#      Prevalence resolves only to 1/N. The shared-generator arms (fedavg,
#      centralized) have no per-site ceiling -- train.py hands each hospital the
#      whole pooled generation -- so they hold 64,000 per hospital where local
#      and fedavg_ft hold 8,000. Uncapped, they score a better R2 for reasons
#      unrelated to their generator. The cap puts all four on one grid.
#
#  test2 --train-budget 8000
#      Every downstream classifier trains on the same number of records, so a
#      TSTR win is a win on data QUALITY rather than on how much data the arm
#      happened to hold. Uncapped answers the other (also valid) question -- can
#      a 142-patient site draw more usable data from a generator than it owns --
#      and that version already exists in test2_rare_efficacy.json.
#
#      8,000 matches test1's --synth-cap, so both axes describe the same amount
#      of synthetic data per hospital. It is also the largest cap that wastes
#      nothing: local and fedavg_ft hold exactly 8,000 per hospital, so they use
#      all of it, while fedavg and centralized come down from 64,000. Total
#      classifier volume is 4x8,000 = 32,000 against the uncapped run's
#      2x16,000 + 2x2,000 = 36,000, so this is no slower than the 5 min that
#      took -- capping here costs nothing and buys the comparison.
#
#      The one arm the cap cannot reach is real_local: a hospital holds what it
#      holds (median 1,259, smallest 142) and no cap can add records. It stays
#      at its true size, which is the honest baseline.
#
# Both write to new files so the earlier scores stay readable.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=rescore-n8000
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_save \
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_save \
      --run centralized=_outputs/centralized_full_hilo8_random_save \
      --run local=_outputs/local_full_hilo8_random_save"

echo "############ Test 1, capped at 8000/hospital"
python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" $RUNS \
    --fold test --synth-cap 8000 \
    --out _outputs/results/tests/test1_prevalence_n8000_capped.json

echo "############ Test 2, matched budget 8000 records/classifier"
python examples/fedpyhealth/test2_rare_efficacy.py \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" $RUNS \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_rare_efficacy_n8000_budget8000.json
