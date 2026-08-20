#!/bin/bash
# Build a cohort cache: the one job that reads eICU. Everything else reads what
# this writes.
#
# Defaults build hilo8_random -- eight hospitals, four at >= 1500 post-task
# patients and four drawn at random from 100-1499, all above the floor where
# "rare" is a real band rather than a single count (see FLOOR below).
#
#   sbatch examples/fedpyhealth/scripts/run_cohort.sh
#
# Everything is overridable from the environment, so a different cohort needs
# no edit to this file:
#
#   COHORT_NAME=big8_random COHORT_HOSPITALS=264,420,243,338,458,443,73,188 \
#       sbatch examples/fedpyhealth/scripts/run_cohort.sh
#   COHORT_SPLIT=stratified sbatch examples/fedpyhealth/scripts/run_cohort.sh
#
# Paths come from the environment too -- so the same script runs on the cluster
# and on a laptop, and no one's personal path is committed:
#
#   export EICU_ROOT=/path/to/eicu-crd/2.0         # required
#   export FEDCOHORT_CACHE=/fast/scratch/fedcohort # optional, see below
#
# Put those in ~/.bashrc once. sbatch forwards your environment by default, so
# a one-off override also works: EICU_ROOT=... sbatch scripts/run_cohort.sh
#
# Rebuilding RE-SPLITS the data: it invalidates every checkpoint and makes
# already-finished runs incomparable to new ones. Build once, then keep
# pointing every train/test job at the same directory. Building a NEW cohort
# under a new COHORT_NAME is safe -- it writes its own directory and leaves the
# existing ones untouched.
#
# Runtime is ~1-2h, almost all of it the single pass over eICU. It is CPU and
# I/O bound; a GPU is requested even though nothing touches one, because Delta
# rejects any zero-GPU job submitted under a *-delta-gpu account and this
# project holds only GPU accounts. gpuA40x4 is the cheapest tier and usually the
# shortest queue -- do not "upgrade" it to A100.
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

NAME="${COHORT_NAME:-hilo8_random}"
SPLIT="${COHORT_SPLIT:-random}"
SEED="${COHORT_SEED:-0}"

# The frozen draw, recorded in cohorts/<NAME>.config.json. Passing the ids
# explicitly (rather than re-drawing from the bands) is what makes this
# reproduce the SAME cohort every time -- draw_bands samples uniformly within
# each band, so a re-draw is a different cohort even at the same seed.
#
# Set COHORT_HOSPITALS='' to draw fresh from the bands instead. The build
# prints the draw and a `--hospitals ...` line; pin that line here and record it
# under cohorts/ so the cache can be rebuilt if it is ever lost.
HOSPITALS="${COHORT_HOSPITALS-458,188,300,208,449,277,358,429}"

# FLOOR=100: a hospital's rare codes are those held by >= 2 patients but
# <= 5% of them, so below 40 patients no code can be both and the build dies
# with "empty rare set". 40 is degenerate -- "rare" would mean exactly 2
# patients, sitting on the 5% line. At 100 the band is 2-5 patients, the first
# size where it has real width. 1500 splits the eight into four large sites and
# four drawn from the long tail.
BANDS="${COHORT_BANDS:-100-1499,1500-}"
PER_BAND="${COHORT_PER_BAND:-4}"

if [ -z "${EICU_ROOT:-}" ]; then
    echo "EICU_ROOT is not set. Point it at the folder holding patient.csv:" >&2
    echo "  export EICU_ROOT=/path/to/eicu-crd/2.0" >&2
    exit 1
fi
# Defaults to a repo-relative gitignored dir so a fresh clone works untouched;
# on a cluster point FEDCOHORT_CACHE at fast local storage instead.
ROOT="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}"
OUT="$ROOT/$NAME"

echo "eICU:   $EICU_ROOT"
echo "cache:  $OUT"
echo "split:  $SPLIT   seed: $SEED"

# A half-written cache from a failed build has parquet files but no
# manifest.json, so nothing downstream can load it -- but its stale per-hospital
# parquet would survive a rebuild that draws different hospitals. Refuse rather
# than silently mix two draws in one directory.
if [ -d "$OUT" ] && [ ! -f "$OUT/manifest.json" ]; then
    echo "$OUT exists but has no manifest.json -- a previous build failed part" >&2
    echo "way through. Remove it first:  rm -rf $OUT" >&2
    exit 1
fi

source .venv/bin/activate

COHORT=examples/fedpyhealth/utils/cohort.py

if [ -n "$HOSPITALS" ]; then
    echo "draw:   pinned ($HOSPITALS)"
    SELECT=(--hospitals "$HOSPITALS")
else
    echo "draw:   $PER_BAND per band from $BANDS (seed $SEED)"
    SELECT=(--bands "$BANDS" --per-band "$PER_BAND")
fi

echo
echo "==================== building $NAME (--split $SPLIT) ===================="
python "$COHORT" \
    "${SELECT[@]}" \
    --out "$OUT" \
    --name "$NAME" \
    --split "$SPLIT" \
    --seed "$SEED" \
    "$@"

# The build verified itself above (rebuild from Parquet, compare tensors).
# This is the fold coverage you actually read.
echo
echo "==================== report: $NAME ===================="
python "$COHORT" --report --out "$OUT"

echo
echo "Next:  python examples/fedpyhealth/main.py all --profile full --cohort-cache $OUT"
