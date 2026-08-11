#!/bin/bash
# Build the cohort caches: the one job that reads eICU. Everything else reads
# what this writes.
#
# Builds TWO caches over the SAME eight hospitals, differing only in how each
# hospital's patients are split into train/val/test:
#
#   strat8         iterative multilabel stratification over rare codes, with
#                  >=1 patient of every rare code guaranteed in train and test
#   strat8_random  a plain seeded 70/10/20 shuffle, blind to rare codes
#
# The pair is the experiment: run `--report` on both and the difference tells
# you what the stratification is actually buying, rather than assuming it.
#
# Paths come from the environment, not from this file -- so the same script
# runs on the cluster and on a laptop, and no one's personal path is committed:
#
#   export EICU_ROOT=/path/to/eicu-crd/2.0        # required
#   export FEDCOHORT_CACHE=/fast/scratch/fedcohort # optional, see below
#
# Put those in ~/.bashrc once. sbatch forwards your environment by default, so
# a one-off override also works: EICU_ROOT=... sbatch scripts/run_cohort.sh
#
# Run ONCE from the repo root:
#   mkdir -p _outputs/slurm
#   sbatch examples/fedpyhealth/scripts/run_cohort.sh
#
# Rebuilding RE-SPLITS the data: it invalidates every checkpoint and makes
# already-finished runs incomparable to new ones. Build once, then keep
# pointing every train/test job at the same directory.
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
# from --bands) is what makes this reproduce the SAME cohort every time.
HOSPITALS="420,199,345,79,259,253,438,201"
SEED=0

if [ -z "${EICU_ROOT:-}" ]; then
    echo "EICU_ROOT is not set. Point it at the folder holding patient.csv:" >&2
    echo "  export EICU_ROOT=/path/to/eicu-crd/2.0" >&2
    exit 1
fi
# Defaults to a repo-relative gitignored dir so a fresh clone works untouched;
# on a cluster point FEDCOHORT_CACHE at fast local storage instead.
ROOT="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}"
echo "eICU:  $EICU_ROOT"
echo "cache: $ROOT"

source .venv/bin/activate

COHORT=examples/fedpyhealth/utils/cohort.py

# Same hospitals, same seed, same eICU -- only --split differs, so any
# difference in the reports is attributable to the splitter alone. The second
# build re-reads eICU, but PyHealth's processed-sample cache makes that cheap.
for SPLIT in stratified random; do
    if [ "$SPLIT" = "stratified" ]; then OUT="$ROOT/strat8"; NAME=strat8
    else OUT="$ROOT/strat8_random"; NAME=strat8_random; fi

    echo
    echo "==================== building $NAME (--split $SPLIT) ===================="
    python "$COHORT" \
        --hospitals "$HOSPITALS" \
        --out "$OUT" \
        --name "$NAME" \
        --split "$SPLIT" \
        --seed "$SEED" \
        "$@"
done

# Both caches verified themselves above (rebuild from Parquet, compare tensors).
# Now the comparison you actually want.
for NAME in strat8 strat8_random; do
    echo
    echo "==================== report: $NAME ===================="
    python "$COHORT" --report --out "$ROOT/$NAME"
done

echo
echo "Next:  python examples/fedpyhealth/main.py all --profile full --cohort-cache $ROOT/strat8"
