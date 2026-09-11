#!/bin/bash
# Regenerate and score the dropout-0.1 XM arms, plus the dropout-0.3 K=1 control
# that the earlier chained job skipped (its path has no _xm1 suffix, and the
# fix landed after sbatch had already snapshotted the script).
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=score-xm01
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
C=_outputs/centralized_E2_R50_hilo8_random_privacy_nes
F=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes

RUNS=""
regen () {
    [ -d "$2" ] || { echo "missing $2" >&2; return; }
    echo "############ regenerating $1"
    python examples/fedpyhealth/generate.py --save-dir "$2" \
        --cohort-cache "$CACHE" --synth-per-hospital 8000
    RUNS="$RUNS --run $1=$2"
}
regen do03_only     "${C}_do0.3_save"
regen do01_only     "${C}_do0.1_save"
regen do01_cent_k4  "${C}_do0.1_xm4_save"
regen do01_fed_k4   "${F}_do0.1_xm4_save"
regen do01_fed_irm  "${F}_do0.1_xm4_irm100w17_save"
[ -n "$RUNS" ] || { echo "nothing to score" >&2; exit 1; }

for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "_outputs/results/tests/test1_xm01_${tag}_b200.json"
done
