#!/bin/bash
# DP-FedAvg STAGE 1: measure client update norms. No noise, no real clipping.
#
# The clip norm S is the one DP knob that cannot be guessed. Too high and the
# noise (scaled by S) drowns the signal; too low and every update is truncated
# and the model stops moving. The standard choice is the median update norm,
# which has to be MEASURED first -- so this run sets --dp-clip absurdly high
# (nothing clips) and --dp-sigma 0 (nothing is noised), purely to print the
# per-round norms from the [dp] log lines.
#
# BUDGET. --local-epochs 10 --n-rounds 10 = 100 passes/hospital, identical to
# the v2 arms, but TEN aggregations instead of fifty. Under DP-FedAvg the
# privacy cost is one Gaussian release per ROUND, so going 50 -> 10 rounds cuts
# epsilon by roughly half at fixed sigma (see utils/dpaccount.py) and costs no
# compute at all. This is the one place "fewer synchronisations" genuinely buys
# privacy rather than just utility.
#
# UNIFORM WEIGHTING IS MANDATORY, not a preference. With sample-count weights
# the largest site moves the mean by more than clip/n, so the noise scale would
# understate the sensitivity and the reported epsilon would be too small.
# utils/dpfed.py refuses rather than silently under-noising -- which also means
# these runs are NOT directly comparable to the sample-weighted v2 arms, and a
# uniform non-DP control at the same budget is needed. That is task 2 of this
# array.
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dp-probe
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
BASE="--profile full --cohort-cache $FEDCOHORT_CACHE/hilo8_random
      --local-epochs 10 --n-rounds 10 --metrics privacy --no-early-stop
      --weighting uniform"

case "$SLURM_ARRAY_TASK_ID" in
 # 1e9 clip = measurement only: nothing is clipped, nothing is noised.
 0) NAME=dp_probe_fedavg;   ARGS="--regime fedavg --dp-clip 1e9 --dp-sigma 0" ;;
 1) NAME=dp_probe_irmxm4;   ARGS="--regime fedavg --dp-clip 1e9 --dp-sigma 0
                                  --irm-rho 100 --irm-warmup 17 --dropout 0.1 --xm-k 4" ;;
 # The control the DP arms are read against: same budget, same 10 rounds, same
 # uniform weighting, no DP path at all.
 2) NAME=dp_ctrl_fedavg;    ARGS="--regime fedavg" ;;
 *) echo "bad array index" >&2; exit 1 ;;
esac

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py $BASE $ARGS --run-name "$NAME" --resume
echo done
