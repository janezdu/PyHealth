#!/bin/bash
# Test 2 for the ouroboros arms on the TAIL pool -- the 478 rare codes with
# 4-fold masking, which is what the visualisation's "All codes" / "Rare codes"
# tabs show. The sweep scorer runs only the head pool, so these four arms
# currently appear on one tab of three.
#
# Worth knowing what this pool is: half its codes carry 1-4 test positives, the
# real-data control real_local scores AUROC 0.40 on it, and nothing separates
# arms. It is here for CONTINUITY with the 22 arms already plotted, not because
# it measures better than the head pool.
#
# Anchors rescored alongside so the four land in one file with a reference.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-ouro-tail
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

RUNS=""
while IFS='=' read -r arm dir; do
    [ -n "${arm:-}" ] && [ -f "$dir/synthetic.json" ] && RUNS="$RUNS --run $arm=$dir"
done < _outputs/results/ouro2_manifest.txt
RUNS="$RUNS --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save"
RUNS="$RUNS --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save"
echo "scoring:$RUNS"

python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_ouro_tail_budget8000.json
echo done
