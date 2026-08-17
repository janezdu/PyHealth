#!/bin/bash
# Test 1, all four regimes, matched synthetic sample size.
#
#   sbatch examples/fedpyhealth/scripts/test1_4arm.sbatch
#
# No GPU: prevalence is pandas and numpy. Fine on an interactive node too --
#   srun --account=bgyw-delta-gpu --partition=gpuA100x4 --gpus-per-node=1 \
#        --mem=64g --time=00:30:00 --pty bash
# then run the python line below directly.
#
# --synth-cap 2000 is the point of this script. centralized and fedavg hold
# 16000 synthetic patients per hospital, local and fedavg_ft hold 2000. A
# synthetic set of size N can only express prevalences in multiples of 1/N, so
# scoring 16000 against 2000 hands the single-model arms a finer grid and a
# better R^2 for a reason that has nothing to do with generator fidelity --
# and it flatters exactly the arms the hypothesis expects to win. Capping every
# arm at 2000 makes the four-way comparison about the generators.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-test1-4arm
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail

source .venv/bin/activate

python examples/fedpyhealth/test1_prevalence.py \
    --run centralized=_outputs/centralized_full_strat8_random_save \
    --run fedavg=_outputs/fedavg_full_strat8_random_save \
    --run fedavg_ft=_outputs/fedavg_ft_full_strat8_random_save \
    --run local=_outputs/local_full_strat8_random_save \
    --synth-cap 2000 \
    --n-bootstraps 10 \
    --out _outputs/results/tests/test1_4arm_cap2000.json \
    "$@"
