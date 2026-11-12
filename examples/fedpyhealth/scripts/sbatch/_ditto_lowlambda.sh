#!/bin/bash
# Ditto at lambda = 0.01 and 0.001, extending the sweep downward.
#
# WHY. The existing sweep {0, 0.1, 1, 10} shows lambda is a shrinkage dial that
# barely turns: 0.1 -> 1 -> 10 is a hundredfold change in the proximal weight
# and moves specialist R^2 by 0.04 (+0.268 -> +0.226). Nearly all of Ditto's
# range sits between lambda=0 and lambda=0.1 -- and lambda=0 removes the
# proximal term entirely, so it IS plain FedAvg+FT, not Ditto.
#
# That leaves the interesting region unsampled. These two fill it: if the
# baseline's frontier segment really runs from (+0.515, +0.620) at lambda=0 to
# (+0.268, +0.852) at lambda=0.1, then 0.01 and 0.001 should land between them
# and trace the line the paper claims these methods are confined to. If instead
# they jump straight to the lambda=0 corner, the dial is effectively binary and
# the "baselines trace a line" framing needs rewording.
#
# Every knob copied from the existing lambda=0.1 run so the five points differ
# in lambda alone: E2, R50, ft2, no early stopping, sample weighting, adapter
# none, batch 256.
#
# make_run_name encodes lambda, so these land in their own save_dirs
# (ditto_E2_R50_..._ditto0.01_nes) and cannot clobber the existing sweep.
#
# ~15-25 min each based on the earlier Ditto runs (bl-ditto0 15m, bl-ditto1
# 24m, bl-ditto10 19m); 2h is margin, and generation plus scoring run in the
# same task so each arm arrives complete.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ditto-low
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

LAMBDAS=(0.01 0.001)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
NAME="ditto${LAM}"

echo "### Ditto lambda=$LAM  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py --profile full --cohort-cache "$CACHE" \
    --local-epochs 2 --n-rounds 50 --metrics privacy --no-early-stop \
    --regime ditto --ft-epochs 2 --ditto-lambda "$LAM" --resume

# train.py names it itself; ask rather than guess, so a rename upstream cannot
# silently point the scoring at the wrong directory.
D=$(python examples/fedpyhealth/train.py --profile full --cohort-cache "$CACHE" \
      --local-epochs 2 --n-rounds 50 --metrics privacy --no-early-stop \
      --regime ditto --ft-epochs 2 --ditto-lambda "$LAM" --print-save-dir | tail -1)
echo "### save_dir = $D"

echo "### generating at 8000/hospital"
python examples/fedpyhealth/generate.py --save-dir "$D" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="_outputs/results/tests/test1_${NAME}_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" \
        --run "$NAME=$D" --fold test --synth-cap 8000 --rare-scope pooled \
        --real-scope "$scope" --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done
echo done
