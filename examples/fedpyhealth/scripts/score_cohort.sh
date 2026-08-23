#!/bin/bash
# Score a trained cohort: the standard, controlled evaluation every experiment
# gets. Run this after `main.py all` has finished training the four regimes.
#
#   COHORT=lo8_random sbatch examples/fedpyhealth/scripts/score_cohort.sh
#   COHORT=hilo8_random PER_HOSP=8000 sbatch examples/fedpyhealth/scripts/score_cohort.sh
#
# `main.py all` trains and then scores ONCE, with every control off. Those
# numbers are fine for a smoke check and misleading as a comparison between
# arms. This script is what produces publishable numbers. See
# notes/RECIPE.md for the reasoning behind each control.
#
# WHAT IT RUNS  (4 steps, ~30-60 min on one A100)
#
#   1. Regenerate synthetic at PER_HOSP per hospital, from the checkpoints
#      already on disk. No retraining.
#   2. Test 1, SPECIALIST target -- each site's synthetic against its OWN test
#      fold (--real-scope hospital).
#   3. Test 1, GENERALIST target -- every site's synthetic against the SAME
#      pooled cohort test fold (--real-scope pooled).
#   4. Test 2, matched budget -- every downstream classifier trains on PER_HOSP
#      records.
#
# WHY THE CONTROLS
#
#   --synth-cap PER_HOSP  (steps 2-3)
#       Prevalence resolves only to 1/N. The shared-generator arms (fedavg,
#       centralized) have no per-site ceiling -- train.py hands each hospital
#       the whole pooled generation -- so uncapped they hold 8x what local and
#       fedavg_ft hold and score a better R2 for reasons unrelated to their
#       generator. The cap puts all four on one grid.
#
#   --train-budget PER_HOSP  (step 4)
#       Same problem on the utility axis. Uncapped, lo8's centralized scored
#       AP 0.0209 against real_pooled's 0.0192 -- synthetic beating real, which
#       is 16,000 classifier records against 2,684, not a good generator.
#       The one arm the cap cannot reach is real_local: a hospital holds what it
#       holds and no cap adds records. It stays at its true size, which is the
#       honest baseline.
#
#   --rare-scope pooled  (steps 2-3)
#       One shared definition of "rare" -- the union of codes rare at ANY site.
#       Per-hospital scope gives each site a different tail (296 codes at one
#       hilo8 site, 103 at another), so those numbers are not comparable to each
#       other or across the two targets.
#
#   BOTH real scopes, always
#       They answer different questions and can rank the arms differently:
#       "is this generator a good specialist for its own site" vs "does it match
#       the cohort". Reporting one alone presents a choice of target as a fact
#       about the generators. The dashboard's head-vs-tail panel needs both.
#
# WHY PER_HOSP=8000 AND NOT THE COHORT'S OWN SIZE
#       It is held FIXED across cohorts on purpose. hilo8_random was regenerated
#       and capped at 8,000, so lo8_random uses 8,000 too -- same 1/8000
#       prevalence resolution, same classifier volume, one axis. Scaling it to
#       each cohort's size would save a little GPU time and cost every
#       cross-cohort comparison. Change it only for a deliberate resolution
#       study, and then change it for every cohort you mean to compare.
#
# Every step writes its own file, named for the cohort. Nothing here overwrites
# another cohort's results -- that is a mistake this project has already made.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=score-cohort
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail

COHORT="${COHORT:-hilo8_random}"
PER_HOSP="${PER_HOSP:-8000}"
PROFILE="${PROFILE:-full}"
ARMS="${ARMS:-fedavg fedavg_ft centralized local}"
# Suffix for a variant condition (e.g. SUFFIX=_rw for the rare-upweighted runs),
# so a variant never lands on the baseline's filenames.
SUFFIX="${SUFFIX:-}"

ROOT="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}"
CACHE="$ROOT/$COHORT"
RESULTS=_outputs/results/tests
STEM="${COHORT}${SUFFIX}"

if [ ! -f "$CACHE/manifest.json" ]; then
    echo "no cohort cache at $CACHE -- build it first:" >&2
    echo "  python examples/fedpyhealth/utils/cohort.py --name $COHORT --out $CACHE ..." >&2
    exit 1
fi

source .venv/bin/activate
mkdir -p "$RESULTS"

RUNS=""
for a in $ARMS; do
    d="_outputs/${a}_${PROFILE}_${COHORT}${SUFFIX}_save"
    if [ ! -d "$d" ]; then
        echo "missing run directory $d -- train it first:" >&2
        echo "  python examples/fedpyhealth/main.py all --profile $PROFILE --cohort-cache $CACHE" >&2
        exit 1
    fi
    RUNS="$RUNS --run ${a}=${d}"
done

echo "cohort:   $COHORT   ($CACHE)"
echo "arms:     $ARMS"
echo "per-hosp: $PER_HOSP"
echo

# Keep whatever synthetic set produced the previous scores. generate.py
# overwrites synthetic.json in place, so without this any number already
# reported from these runs becomes unreproducible. Guarded, so re-running does
# not overwrite the backup with the new version.
for a in $ARMS; do
    d="_outputs/${a}_${PROFILE}_${COHORT}${SUFFIX}_save"
    if [ -f "$d/synthetic.json" ] && [ ! -f "$d/synthetic_prev.json" ]; then
        cp "$d/synthetic.json" "$d/synthetic_prev.json"
        echo "backed up $d/synthetic.json -> synthetic_prev.json"
    fi
done

echo "############ 1/4  regenerating at ${PER_HOSP}/hospital (no retraining)"
for a in $ARMS; do
    echo "---- $a"
    python examples/fedpyhealth/generate.py \
        --save-dir "_outputs/${a}_${PROFILE}_${COHORT}${SUFFIX}_save" \
        --cohort-cache "$CACHE" \
        --synth-per-hospital "$PER_HOSP"
done

echo "############ 2/4  Test 1, SPECIALIST target (each site vs its own fold)"
python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap "$PER_HOSP" \
    --rare-scope pooled --real-scope hospital \
    --out "$RESULTS/test1_prevalence_${STEM}_capped_pooledrare.json"

echo "############ 3/4  Test 1, GENERALIST target (all sites vs pooled fold)"
python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap "$PER_HOSP" \
    --rare-scope pooled --real-scope pooled \
    --out "$RESULTS/test1_prevalence_${STEM}_pooledreal_pooledrare.json"

echo "############ 4/4  Test 2, matched budget ${PER_HOSP} records/classifier"
python examples/fedpyhealth/test2_rare_efficacy.py \
    --cohort-cache "$CACHE" $RUNS \
    --fold test --train-budget "$PER_HOSP" \
    --out "$RESULTS/test2_rare_efficacy_${STEM}_budget${PER_HOSP}.json"

echo
echo "Done. Render the dashboard with a config naming these three files:"
echo "  python examples/fedpyhealth/eda.py fidelity --config examples/fedpyhealth/viz/${COHORT%%_*}.yaml"
