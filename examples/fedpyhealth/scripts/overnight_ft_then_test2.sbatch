#!/bin/bash
# Overnight: finish fedavg_ft, then score the three arms that will be ready.
#
#   sbatch examples/fedpyhealth/scripts/overnight_ft_then_test2.sbatch
#
# Two steps in ONE allocation rather than two jobs chained with --dependency.
# A dependency that never satisfies leaves the child parked as
# DependencyNeverSatisfied until someone notices and scancels it, which is a bad
# thing to discover in the morning. `set -e` gives the same safety here: if the
# fine-tuning step fails, the job stops and test2 never runs on stale data.
#
# Step 1  re-run fedavg_ft's fine-tuning half so ft_<hosp>.pt lands on disk and
#         its per-hospital sets are regenerated at 2000 each. --resume skips the
#         49 completed FedAvg rounds, so this is ~40 min, not ~19h.
# Step 2  test2 over centralized + fedavg + fedavg_ft, uncapped, 4 mask folds.
#
# `local` is deliberately absent: it is still training (~17h out) and would need
# its own generate.py pass afterwards. Re-run test2 with a fourth --run once it
# lands; nothing here has to be redone.
#
# Arms: real_pooled (1) + real_local (8) + tstr:centralized (1)
#     + tstr:fedavg (1) + tstr:fedavg_ft (8) = 19 arms x 4 folds = 76
# classifiers, ~59.6k records per fold. test2 has never run to completion, so
# the time limit is deliberately generous rather than tuned.
#
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-overnight
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail

source .venv/bin/activate

FT_SAVE=_outputs/fedavg_ft_full_strat8_random_save
if [ -f "$FT_SAVE/synthetic.json" ]; then
    cp -n "$FT_SAVE/synthetic.json" "$FT_SAVE/synthetic.json.bak"
    echo "backed up $FT_SAVE/synthetic.json -> .bak"
fi

echo "=============== STEP 1/2: fedavg_ft fine-tuning + regeneration ==============="
date
# --run-name is NOT optional. main.py supplies the run name (run_name_for);
# train.py called directly derives a DIFFERENT one from make_run_name, giving a
# different save_dir, so --resume finds no checkpoint and restarts FedAvg from
# round 0. That is what timed out job 21095176 after 12h at round 33/49.
python examples/fedpyhealth/train.py \
    --regime fedavg_ft \
    --profile full \
    --resume \
    --run-name fedavg_ft_full_strat8_random

# Guard: fine-tuning must actually have produced the per-hospital generators,
# or test2 below would score whatever stale synthetic.json is still on disk.
for h in 420 199 345 79 259 253 438 201; do
    test -f "$FT_SAVE/ft_$h.pt" || {
        echo "MISSING $FT_SAVE/ft_$h.pt -- fine-tuning did not complete; refusing to score." >&2
        exit 1
    }
done
echo "all 8 ft_<hosp>.pt present"

echo
echo "=============== STEP 2/2: test2 over 3 arms ==============="
date
python examples/fedpyhealth/test2_rare_efficacy.py \
    --run centralized=_outputs/centralized_full_strat8_random_save \
    --run fedavg=_outputs/fedavg_full_strat8_random_save \
    --run fedavg_ft=_outputs/fedavg_ft_full_strat8_random_save \
    --mask-folds 4 \
    --train-budget 0 \
    --resume \
    --out _outputs/results/tests/test2_3arm_uncapped.json

echo
date
echo "done"
