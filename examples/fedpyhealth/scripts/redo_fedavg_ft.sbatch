#!/bin/bash
# Re-run fedavg_ft's fine-tuning half, so its 8 personalized generators land on
# disk as ft_<hospital>.pt.
#
#   sbatch examples/fedpyhealth/scripts/redo_fedavg_ft.sbatch
#
# Why this is cheap: --resume loads fedavg_state.pt, sees it already holds all
# 49 completed rounds, and skips the FedAvg half entirely (train.py's run_fedavg
# returns early when start_round >= n_rounds). What actually re-runs is 2
# fine-tuning epochs per hospital, generation, and evaluation -- roughly 40
# minutes against the ~19h the original run took.
#
# Why it is needed at all: finetune_local never persisted its models, so
# fedavg_ft is the one regime whose per-hospital generators existed only in RAM.
# Without this, re-sampling it at a different size is impossible and
# generate.py refuses rather than silently using the pre-fine-tuning global.
#
# This overwrites the run's synthetic.json and results JSON, which is intended:
# the point is to replace the old size-proportional per-hospital sets (68
# patients for hospital 438) with the uniform 4000 the full profile now asks for.
#
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-fedavg_ft-redo
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail

source .venv/bin/activate

SAVE_DIR=_outputs/fedavg_ft_full_strat8_random_save
if [ -f "$SAVE_DIR/synthetic.json" ]; then
    cp -n "$SAVE_DIR/synthetic.json" "$SAVE_DIR/synthetic.json.bak"
    echo "backed up existing synthetic.json -> $SAVE_DIR/synthetic.json.bak"
fi

# --run-name is NOT optional here. main.py computes the run name itself and
# passes it in (run_name_for); train.py called directly falls back to its own
# make_run_name, which produces a DIFFERENT name for the same run. A different
# name means a different save_dir, which means --resume finds no checkpoint and
# silently restarts FedAvg from round 0. That cost 12h on job 21095176.
python examples/fedpyhealth/train.py \
    --regime fedavg_ft \
    --profile full \
    --resume \
    --run-name fedavg_ft_full_strat8_random \
    "$@"
