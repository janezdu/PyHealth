#!/bin/bash
# Batch wrapper for the logreg Test 2 sweep. The work is pure CPU -- sklearn
# plus the parquet/JSON reads -- but Delta rejects zero-GPU jobs under the gpu
# account, so it takes an allocation it does not use. Switch this to a CPU
# account when one exists; only the SBATCH block needs to change.
#
# CUDA stays hidden on purpose: nothing here benefits from it, and pinning it
# off keeps the run identical to the interactive invocation.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-logreg
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
export LOGREG_C="${LOGREG_C:-0.1 1.0 10}"
bash examples/fedpyhealth/scripts/test2_logreg_cpu.sh
