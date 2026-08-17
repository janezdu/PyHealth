#!/bin/bash
# Runtime probe for test2 under the new uncapped budget, on the two arms whose
# checkpoints are already usable.
#
#   sbatch examples/fedpyhealth/scripts/probe_test2.sbatch
#
# Under --train-budget 0 every arm trains on all the data it has, taking the
# per-fold training volume from ~2.1k records to ~53.6k across the full
# four-regime run (at synth_per_hospital=2000). test2 has never once run to
# completion, so this scores 2 regimes first and reports real timings.
#
# Arms here: real_pooled (1) + real_local (8) + tstr:centralized (1)
#          + tstr:fedavg (1) = 11 arms x 4 folds = 44 classifiers,
# against 27 arms x 4 = 108 for the full run. Per-fold record volume is ~21.6k,
# roughly 40% of the full run, so the extrapolation is honest.
#
# centralized and fedavg are single-generator regimes, so their existing
# synthetic.json already works unchanged -- synthetic_arms detects the shared
# set and builds ONE classifier rather than eight slices of one distribution.
#
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-test2-probe
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail

source .venv/bin/activate

python examples/fedpyhealth/test2_rare_efficacy.py \
    --run centralized=_outputs/centralized_full_strat8_random_save \
    --run fedavg=_outputs/fedavg_full_strat8_random_save \
    --mask-folds 4 \
    --train-budget 0 \
    --resume \
    --out _outputs/results/tests/test2_probe_centralized_fedavg.json \
    "$@"
