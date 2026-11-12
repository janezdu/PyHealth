#!/bin/bash
# The frontier figure's six arms, rerun with --rare-upweight.
#
# WHAT --rare-upweight DOES. Each patient's loss is scaled by how rare their
# codes are AT THEIR OWN SITE: w = 1 + sum of 1/local_prevalence over that
# site's rare codes the patient carries, capped at the site's 99th percentile
# and normalised to mean 1. Keyed on patient_id, so even the pooled centralized
# arm weights each patient by their own site's view of rarity.
#
# EARLY STOPPING IS ON HERE, deliberately, matching the original _rw runs -- it
# is train.py's default, so the flag is simply absent. Note what that costs:
# every arm currently on the figure runs --no-early-stop, so an rw arm differs
# from its plotted twin in TWO ways, and a difference between them cannot be
# attributed to the weighting alone. Rerunning either side with the other's
# early-stopping setting is what would separate them.
#
# THREE OF THE SIX ALREADY EXIST at exactly this configuration and are reused
# rather than recomputed -- local, fedavg and centralized, all `full` profile on
# hilo8_random with rw and early stopping on, all with synthetic.json already
# written. Training is seeded, so rerunning them would reproduce the same
# weights; the manifest points at the existing directories instead.
set -euo pipefail

SWEEP="${SWEEP:-rwfig}"
MANIFEST="_outputs/results/${SWEEP}_manifest.txt"
mkdir -p _outputs/results _outputs/slurm
: > "$MANIFEST"
export SWEEP MANIFEST

# Already trained at this exact config; reused.
cat >> "$MANIFEST" <<'MAN'
local=_outputs/local_full_hilo8_random_rw_save
fedavg=_outputs/fedavg_full_hilo8_random_rw_save
centralized=_outputs/centralized_full_hilo8_random_rw_save
MAN
echo "reusing 3 existing _rw runs (local, fedavg, centralized)"

# ES_FLAG="" leaves early stopping at train.py's default, which is ON.
sub () {
    local arm="$1"; local regime="$2"; shift 2
    local id
    id=$(sbatch --parsable --job-name="rw-$arm" \
         --export=ALL,ARM="$arm",ARGS="$*",MANIFEST="$MANIFEST",REGIME="$regime",ES_FLAG= \
         examples/fedpyhealth/scripts/sbatch/_sweep_arm.sh)
    echo "  $id  $arm  ($regime)  <- $*"
    IDS="${IDS:+$IDS:}$id"
}

# The three with no _rw twin. Regime and flags copied from each target's own
# config.json -- note xm4 is CENTRALIZED, not fedavg.
sub fedavg_irm1e4 fedavg      --rare-upweight --irm-rho 10000 --irm-warmup 17
sub xm4           centralized --rare-upweight --xm-k 4 --dropout 0.3
sub do01_fed_irm  fedavg      --rare-upweight --irm-rho 100 --irm-warmup 17 --xm-k 4 --dropout 0.1

score=$(sbatch --parsable --dependency=afterok:"$IDS" \
        --export=ALL,SWEEP="$SWEEP",MANIFEST="$MANIFEST" \
        examples/fedpyhealth/scripts/sbatch/_sweep_score.sh)
echo
echo "scoring job $score"
echo "results -> _outputs/results/tests/test1_${SWEEP}_{spec,gen}_b200.json"
echo
echo "Early stopping is ON for these (train.py default), matching the original"
echo "_rw runs -- so an rw arm differs from its plotted twin in TWO ways."
