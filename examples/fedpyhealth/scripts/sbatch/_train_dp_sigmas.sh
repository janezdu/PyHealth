#!/bin/bash
# DP-FedAvg for real: clip 1.0, three noise levels. Just run it.
#
# Control already on disk: dp_ctrl_fedavg (same budget, same uniform weighting,
# no DP path) -- so every arm here has a like-for-like no-noise reference.
#
# clip = 1.0 is the standard default and sits inside the measured range: the
# probe saw client update norms of 6.8 in round 1 falling to 0.24 by round 10,
# so C=1 truncates the first two rounds and passes everything after. Not
# optimal at either end; it is one number, which is the point.
#
# epsilon at delta=1e-5 over 10 rounds, no subsampling credit (all 8 sites
# participate every round), from utils/dpaccount.py:
#     sigma 0.5 -> eps 50.4      sigma 1.0 -> eps 20.2      sigma 2.0 -> eps 8.8
#
# The signal/noise arithmetic predicts these will be badly degraded -- noise
# per parameter is sigma*C/8 against a per-parameter update of ~C/sqrt(6.2M).
# That prediction is a heuristic and has not been measured on this model, so
# the runs are the check on it, not the other way round. Score them on
# prevalence like any other arm and let the numbers say.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dp-sigmas
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"

SIGMAS=(0.5 1.0 2.0)
S=${SIGMAS[$SLURM_ARRAY_TASK_ID]}

echo "### dp sigma=$S clip=1.0  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py --profile full \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --local-epochs 10 --n-rounds 10 --metrics privacy --no-early-stop \
    --weighting uniform --regime fedavg \
    --dp-clip 1.0 --dp-sigma "$S" \
    --run-name "dp_s${S}" --resume
echo done
