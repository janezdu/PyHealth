#!/bin/bash
# UMAP of every patient in hilo8_random, coloured by hospital.
#
# CPU ONLY and small: ~12,156 patients x ~740 codes, no model, no torch. UMAP
# on a matrix that size is a couple of minutes. Sized at 1h purely for headroom
# on the jaccard neighbour search, which is the slow part.
#
# Writes BOTH styles plus the divergence json. The facet version is the one for
# the paper; the overlay exists for a talk and does not survive a colour-vision
# check (see the script's docstring).
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=hosp-umap
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
FIG=examples/fedpyhealth/scripts/fig_hospital_umap.py

python -c "import umap" 2>/dev/null || pip install --quiet umap-learn

python "$FIG" --cohort-cache "$CACHE" --style facet \
    --out _outputs/results/hospital_umap.png

python "$FIG" --cohort-cache "$CACHE" --style overlay \
    --out _outputs/results/hospital_umap_overlay.png

echo "done: _outputs/results/hospital_umap{,_overlay}.png"
