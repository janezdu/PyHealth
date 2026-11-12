#!/bin/bash
# One figure for the whole toy experiment, replacing the ten separate PNGs.
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=toy-one
#SBATCH --time=00:45:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
python -u examples/fedpyhealth/scripts/toy_fedavg_one_figure.py \
    --out _outputs/results/toy_fedavg_summary.png
