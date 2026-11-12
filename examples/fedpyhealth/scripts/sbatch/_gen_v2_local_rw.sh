#!/bin/bash
# One generation, for the one cell of the upweighting x early-stopping 2x2 that
# does not yet have enough synthetic to score.
#
# train.py writes synthetic at the config's synth_per_hospital (2000 for a
# per-hospital regime), but Test 1 scores at --synth-cap 8000. Eleven of the
# twelve cells already carry 8,000-64,000 per hospital from earlier
# regenerations; v2_local_rw_save carries 2,000. Checked before writing this,
# so nothing else is touched -- regenerating a cell OVERWRITES its
# synthetic.json, and the other eleven are inputs to results already on disk.
#
# Local is eight small per-hospital generators, so this is the cheapest
# generation in the project. 01:00:00 is generous.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gen-v2-local-rw
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
D=_outputs/v2_local_rw_save

[ -d "$D" ] || { echo "missing $D -- did array task 9 finish?" >&2; exit 1; }
echo "### regenerating $D at 8000/hospital"
python examples/fedpyhealth/generate.py --save-dir "$D" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000
echo done
