#!/bin/bash
# Test 2 on well-supported codes, all three seeds. Batch wrapper around
# scripts/test2_headcodes_cpu.sh.
#
# The script itself is CPU-viable (13 classifiers, no CUDA reference anywhere in
# test2_rare_efficacy.py), but Delta rejects zero-GPU jobs under the gpu account
# and a login node reaps it after a couple of CPU-minutes. So: take the GPU the
# scheduler insists on and let it finish in ~5 min instead of ~25.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-head
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
# Let torch use the GPU here -- unset, not empty, so the script's own default
# does not pin it to CPU.
unset CUDA_VISIBLE_DEVICES
export MINPOS="${MINPOS:-20}" NCODES="${NCODES:-30}" SEEDS="${SEEDS:-0 1 2}"
bash examples/fedpyhealth/scripts/test2_headcodes_cpu.sh
