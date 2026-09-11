#!/bin/bash
# Regenerate and score the XM sweep onto the shared panel protocol.
#
# The in-run eval uses n_bootstraps=5 and whatever synth size the profile set
# (16,000/hospital here). Every point on the specialist/generalist panel is
# 8,000/hospital at n=200 seed 0, and mixing the two would fabricate
# differences -- the same arm scored at n=5 vs n=200 moves by up to 0.017 on
# Pearson, six times the n=200 standard error.
#
# Regeneration needs the GPU; the scoring is CPU and can run on an interactive
# node. Kept in one script because the two must use the same synthetic set.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=score-xm
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
B=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3
FED=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save

RUNS=""
regen () {   # regen <display-name> <save-dir>
    [ -d "$2" ] || { echo "missing $2 -- did it finish?" >&2; return; }
    echo "############ regenerating $1 at 8000/hospital"
    python examples/fedpyhealth/generate.py --save-dir "$2" \
        --cohort-cache "$CACHE" --synth-per-hospital 8000
    RUNS="$RUNS --run $1=$2"
}
# K=1 gets NO _xm suffix -- make_run_name only appends it above 1 -- so the
# dropout-only arm lives at ..._do0.3_save. Reconstructing the path by pattern
# would silently skip it, and it is the arm that carries the attribution.
regen "dropout_only" "${B}_save"
for k in 2 4 8; do
    regen "xm${k}" "${B}_xm${k}_save"
done
regen "xm8_fedavg" "$FED"
[ -n "$RUNS" ] || { echo "nothing to score" >&2; exit 1; }

for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    echo "############ Test 1, $tag"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "_outputs/results/tests/test1_xm_${tag}_b200.json"
done
