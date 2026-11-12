#!/bin/bash
# Two attempts at the FedRep / LG-FedAvg over-emission (2.3x / 4.4x the real
# code volume), each at the SAME 100-pass budget as every baseline:
# 50 rounds x 2 local epochs, no early stopping.
#
#   0  fedrep    --fedrep-alternate   the paper's schedule: head against the
#                                      frozen global body, then body against
#                                      the frozen head (1 + 1 epochs/round)
#   1  fedrep    --rate-mu 1e-3       joint training + a mild penalty on the
#   2  lg_fedavg --rate-mu 1e-3       batch's emitted vs observed code volume
#
# The penalty is relative and squared, so at mu=1e-3 against a BCE of ~5e-4 it
# is ~2% of the loss at a 10% rate error and ~2x it at 2x over-emission. The
# per-client log line prints emitted/observed under teacher forcing: if it
# already sits near 1.00 the penalty has nothing to act on, and any
# over-emission is being created at the final merge, not in training.
#
# ~25 min train by analogy with the original runs, ~13 min generation; 2h is
# margin. Train + generate + Test 1 in one task, like _ditto_lowlambda.sh.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=fedrep-fix
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

NAMES=(fedrep_alt fedrep_rate lgfedavg_rate)
EXTRAS=("--regime fedrep --fedrep-alternate"
        "--regime fedrep --rate-mu 0.001"
        "--regime lg_fedavg --rate-mu 0.001")
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
ARGS="--profile full --cohort-cache $CACHE --local-epochs 2 --n-rounds 50 \
      --metrics privacy --no-early-stop ${EXTRAS[$SLURM_ARRAY_TASK_ID]}"

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py $ARGS --resume
D=$(python examples/fedpyhealth/train.py $ARGS --print-save-dir | tail -1)
echo "### save_dir = $D"

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
