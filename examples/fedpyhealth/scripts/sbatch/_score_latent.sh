#!/bin/bash
# Regenerate, then Test 1 AND Test 2 for the three structured-latent arms.
#
# Paths are resolved by asking train.py, never reconstructed -- guessing the
# run-name component order has already silently dropped two arms this week.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=score-latent
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
Z="--profile full --cohort-cache $CACHE --no-early-stop --latent-dim 16 --xm-k 4"

ask () { python examples/fedpyhealth/train.py $Z "$@" --print-save-dir | tail -1; }
D_CENT=$(ask --regime centralized)
D_FED=$(ask --regime fedavg --irm-rho 100 --irm-warmup 17)
D_LORA=$(ask --regime fedavg_ft --ft-epochs 20 --adapter lora_head --resume \
             --irm-rho 100 --irm-warmup 17)

RUNS=""
add () {
    [ -d "$2" ] || { echo "missing $1 -> $2" >&2; return; }
    echo "############ regenerating $1"
    python examples/fedpyhealth/generate.py --save-dir "$2" \
        --cohort-cache "$CACHE" --synth-per-hospital 8000
    RUNS="$RUNS --run $1=$2"
}
add lat_cent     "$D_CENT"
add lat_fed_irm  "$D_FED"
add lat_lora_ft  "$D_LORA"
[ -n "$RUNS" ] || { echo "nothing to score" >&2; exit 1; }

for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    echo "############ Test 1, $tag"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "_outputs/results/tests/test1_latent_${tag}_b200.json"
done

echo "############ Test 2"
python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_latent_budget8000.json
