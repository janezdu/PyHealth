#!/bin/bash
# The toy: does averaging two generators' weights average their distributions?
# Pure CPU, about a minute. Takes a GPU only because Delta rejects zero-GPU jobs
# under the gpu account.
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=toy-fedavg-stages
#SBATCH --time=00:45:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
python -u examples/fedpyhealth/scripts/toy_fedavg_distribution.py \
    --out _outputs/results/toy_fedavg_stages --stages 0,1,4,10,20
