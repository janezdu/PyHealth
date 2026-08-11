#!/bin/bash
# Build the cohort cache: the one job that reads eICU. Everything else reads
# the cache this writes.
#
# Run ONCE from the repo root:
#   mkdir -p _outputs/slurm
#   sbatch examples/fedpyhealth/scripts/run_cohort.sh
#
# Rebuilding RE-SPLITS the data: it invalidates every checkpoint and makes
# already-finished runs incomparable to new ones. Build it once, then keep
# pointing every train/test job at the same directory.
#
# Overrides pass straight through to utils/cohort.py:
#   sbatch examples/fedpyhealth/scripts/run_cohort.sh --out /work/nvme/.../other
#   sbatch examples/fedpyhealth/scripts/run_cohort.sh --bands 0-499,500- --per-band 4
#
# A GPU is requested even though this job never touches one: Delta rejects any
# zero-GPU job submitted under a *-delta-gpu account, and this project holds
# only GPU accounts. gpuA40x4 is the cheapest tier and usually the shortest
# queue -- do not "upgrade" it to A100.
#
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-cohort-cache
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out

set -euo pipefail

# The frozen draw: two hospitals from each of four size bands, recorded in
# cohorts/strat8.cohort.json. Passing them explicitly (rather than re-drawing
# from --bands) is what makes this script reproduce the SAME cohort every time.
HOSPITALS="420,199,345,79,259,253,438,201"
OUT="/work/nvme/bgyw/janezdu/cache/fedcohort/strat8"

source .venv/bin/activate

python examples/fedpyhealth/utils/cohort.py \
    --hospitals "$HOSPITALS" \
    --out "$OUT" \
    --name strat8 \
    "$@"

# The script verifies itself: it rebuilds every fold from the Parquet files it
# just wrote and fails loudly if a single tensor differs from what the eICU
# path produced. A cache that is merely plausible is worse than no cache.
echo
echo "Next:  python examples/fedpyhealth/main.py all --profile full --cohort-cache $OUT"
