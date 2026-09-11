#!/bin/bash
# Validate self-training before spending anything on it.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/smoke_selftrain.sbatch
#
# Each round, after the server broadcast, every client generates a fraction of
# its own train size from the just-broadcast GLOBAL and trains the local step on
# real + synthetic together. What makes that worth trying is not the extra
# volume -- it is that a 142-patient site is mixing in data shaped by the other
# seven hospitals, with no raw record leaving anyone's machine.
#
# WHAT TO READ IN THE OUTPUT. Every client prints, per round:
#
#   +N/M synthetic (P% non-empty)   generation actually produced patients. A low
#                                   P early is expected -- an untrained global
#                                   emits empty sequences -- and is the argument
#                                   for --selftrain-start.
#   D distinct codes                THE TAIL-COLLAPSE DETECTOR. Feeding a model
#                                   its own output narrows the distribution and
#                                   the tails go first, which on this cohort is
#                                   exactly what is being measured. A monotone
#                                   fall across rounds is collapse in progress
#                                   and shows up long before Test 1 moves.
#   C codes/visit                   real is 11.50; the centralized failure mode
#                                   was 25.17, so drift here is the same signal.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-selftrain
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

echo "############ 1/2  off by default -- must be bit-unchanged"
python examples/fedpyhealth/train.py --profile tiny --regime fedavg \
    --cohort-cache "$CACHE" --run-name smoke_st_off --no-tb 2>&1 \
    | grep -E "synthetic|Saved results|Traceback|Error" | head -5
echo "  (no 'synthetic' lines above = correctly off)"

echo
echo "############ 2/2  25% synthetic per client per round"
python examples/fedpyhealth/train.py --profile tiny --regime fedavg \
    --cohort-cache "$CACHE" --selftrain-frac 0.25 \
    --run-name smoke_st25 --no-tb 2>&1 \
    | grep -E "synthetic|Saved results|Traceback|Error" | head -30
