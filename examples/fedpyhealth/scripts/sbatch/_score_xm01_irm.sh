#!/bin/bash
# Score the fedavg + IRM + XM arm the previous job missed.
#
# It was missed because the scoring script GUESSED the run name and got the
# component order wrong (make_run_name emits _irm100w17 before _nes, not after).
# train.py can be asked directly, which is what --print-save-dir exists for, so
# this resolves the path instead of reconstructing it.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=score-xmirm
#SBATCH --time=01:30:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

D=$(python examples/fedpyhealth/train.py --profile full --regime fedavg \
      --cohort-cache "$CACHE" --no-early-stop --dropout 0.1 --xm-k 4 \
      --irm-rho 100 --irm-warmup 17 --print-save-dir | tail -1)
echo "resolved: $D"
[ -d "$D" ] || { echo "no such dir" >&2; exit 1; }

python examples/fedpyhealth/generate.py --save-dir "$D" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000
for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" \
        --run do01_fed_irm="$D" \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "_outputs/results/tests/test1_xmirm_${tag}_b200.json"
done
