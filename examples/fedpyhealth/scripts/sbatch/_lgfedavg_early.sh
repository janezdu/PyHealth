#!/bin/bash
# LG-FedAvg, stopped early. Its validation loss bottoms at round 12 (0.000458)
# and drifts to 0.000618 by round 50 -- +34.8%. No other baseline does this:
# FedRep drifts 2.2%, FedProx not at all. So its catastrophic prevalence R2
# (-5.1 specialist, -17.7 generalist) may be an overfitting artefact of running
# the full budget rather than a property of the method.
#
# THE ROUND-12 CHECKPOINT DOES NOT EXIST. snapshot_every defaults to 0 ("keep
# no numbered snapshots") and the original run used the default, so every round
# overwrote the last and only the final state survives. This retrains.
#
# NOT the same model. FedAvg is not bit-reproducible across runs, so task 0 is
# "LG-FedAvg trained for 12 rounds", not "that run's round-12 state".
#
# Task 0 -- 12 rounds flat. Answers the question as asked, but spends 24 passes
#           per hospital against 100 for every other arm, so it is NOT
#           compute-matched and cannot go on the frontier figure as a peer.
# Task 1 -- full 50-round budget with early stopping ON. Compute-matched in
#           intent (same budget offered, stops when val stops improving, ~17
#           rounds on this curve) and the version that belongs in a comparison.
#
# --snapshot-every 2 on both, so the next time this question comes up the
# checkpoints are already there.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=lg-early
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
BASE="--profile full --cohort-cache $CACHE --local-epochs 2 --metrics privacy
      --regime lg_fedavg --snapshot-every 2"

case "$SLURM_ARRAY_TASK_ID" in
 0) NAME=lg_fedavg_r12;  ARGS="--n-rounds 12 --no-early-stop" ;;
 1) NAME=lg_fedavg_es;   ARGS="--n-rounds 50" ;;   # early stopping left ON
 *) exit 1 ;;
esac

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py $BASE $ARGS --run-name "$NAME" --resume

D=_outputs/${NAME}_save
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
