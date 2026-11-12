#!/bin/bash
# Smoke for --fedrep-alternate and --rate-mu, tiny profile. Both paths are new
# and both can fail silently:
#   alternate: if the key/parameter names ever disagree, a block trains nothing
#              (train_model raises "no trainable parameters" -- loud, good)
#   rate:      the per-client log line must show "emitted/observed" numbers
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=fedrep-fix-smoke
#SBATCH --time=00:30:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
BASE="--profile tiny --cohort-cache $CACHE --no-tb --no-early-stop --n-rounds 2 --local-epochs 2"
for EXTRA in "--regime fedrep --fedrep-alternate" "--regime lg_fedavg --rate-mu 0.001"; do
    D=$(python examples/fedpyhealth/train.py $BASE $EXTRA --print-save-dir | tail -1)
    case "$D" in _outputs/?*) rm -rf "$D" ;; *) echo "bad save_dir '$D'" >&2; exit 1 ;; esac
    echo "### $EXTRA -> $D"
    python examples/fedpyhealth/train.py $BASE $EXTRA
done
echo SMOKE_OK
