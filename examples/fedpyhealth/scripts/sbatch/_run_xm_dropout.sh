#!/bin/bash
# Best-of-K exploration over dropout masks, centralized.
#
# HALO shipped with NO dropout, so its forward pass was deterministic and an
# earlier best-of-K probe came back with spread=0.000: the K candidates were
# bit-identical, min() picked arbitrarily, and the run was quietly training on
# 1/K of its gradient. Dropout is added here purely to supply the stochasticity
# that XM needs.
#
# READ THE FIRST EPOCH. Every epoch prints
#     xm K=8  spread=...  win_entropy=...
# spread -> 0 or win_entropy -> 0 means the candidates are not distinct and the
# whole run is meaningless. That is the only thing worth checking before letting
# it finish.
#
# WHAT THIS CANNOT SEPARATE. There is no dropout-only arm, so a difference
# against the existing centralized baseline is dropout AND best-of-K together.
# Dropout is a regulariser in its own right; attributing any gain to XM alone
# would need that third arm.
#
# Also: dropout noise is UNSTRUCTURED -- it perturbs the computation, not the
# hypothesis. Best-of-K over dropout masks is closer to "which noise
# realisation happened to fit" than "which phenotype", so a null here is weaker
# evidence against XM than a null with a structured latent would be.
#
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=xm-dropout
#SBATCH --time=06:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

python examples/fedpyhealth/train.py --profile full --regime centralized \
    --cohort-cache "$CACHE" --no-early-stop \
    --dropout 0.3 --xm-k 8
