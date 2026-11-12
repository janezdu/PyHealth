#!/bin/bash
# Score every arm listed in $MANIFEST, plus the two anchors the sweep is read
# against. Runs after all arm jobs finish (submitted with --dependency).
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sweep-score
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS=""
while IFS='=' read -r arm dir; do
    [ -n "${arm:-}" ] || continue
    if [ -f "$dir/synthetic.json" ]; then RUNS="$RUNS --run $arm=$dir"
    else echo "SKIP $arm -- no synthetic.json at $dir" >&2; fi
done < "$MANIFEST"
# Anchors: the centralized best-of-K pair every federated number is read
# against, rescored here so the comparison lives in ONE file.
#
# Prefixed ref: because a sweep may well contain an arm called xm4 itself -- the
# reweighted sweep did -- and two --run entries sharing a name silently collapse
# to one, leaving no way to tell from the output which checkpoint was scored.
add_ref () {
    if grep -q "^$1=" "$MANIFEST"; then
        echo "  anchor $1 renamed to ref_$1 (the sweep has its own $1)" >&2
        RUNS="$RUNS --run ref_$1=$2"
    else
        RUNS="$RUNS --run $1=$2"
    fi
}
add_ref xm4 _outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
add_ref do03_only _outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save
echo "scoring:$RUNS"

for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "$RES/test1_${SWEEP}_${tag}_b200.json"
done

for seed in 0 1 2; do
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" --train-budget 8000 \
        --out "$RES/test2_head_${SWEEP}_seed${seed}.json"
done
echo done
