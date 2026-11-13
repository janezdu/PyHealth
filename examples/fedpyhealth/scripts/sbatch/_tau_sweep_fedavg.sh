#!/bin/bash
# RS1V's Q1: "Does the IRM gain remain if FedAvg and the other baselines get a
# simple calibration step, such as tuning the sampling temperature per
# hospital?"
#
# WHY THIS IS THE SHARPEST OBJECTION IN THE BATCH. Prevalence R^2 is the
# headline metric and it measures emission RATE. The codes are conditionally
# independent Bernoullis, so expected codes/visit is the sum of their
# probabilities, and temperature scales every one of them at once -- a single
# scalar acting directly on the quantity being scored, with no retraining.
#
# Table 1 says the arms differ almost entirely in rate: Pearson spans 0.06
# across every federated arm while R^2 spans -5.1 to +0.72. Pearson sees
# ranking, R^2 sees ranking AND rate. So the separation the paper reports is a
# rate separation, and temperature is a rate knob. If one scalar closes it,
# the invariance story does not survive.
#
# PREDICTION WORTH CHECKING IN THE OUTPUT: temperature is monotone per code,
# so it cannot reorder which codes are common. It should move R^2 a lot and
# Pearson almost not at all. Seeing that confirms the mechanism is rate rather
# than representation, whichever way the headline lands.
#
# DESIGN. One GLOBAL tau rather than eight per-hospital ones, for this first
# pass: it is cheaper, and a single value closing the gap is the more damaging
# -- and more interesting -- result. Per-hospital tau needs records mixed
# across tau files and can follow if the global sweep shows movement.
#
# Five values, 0.8 to 1.2. tau=1.0 is the trained model and already on disk,
# so only four generations are new. Selection is on the VAL fold and reporting
# is on TEST; both are scored here so the choice is visible rather than buried.
#
# generate.py writes a swept tau to synthetic_tau<v>.json so the tau=1 set
# every existing score was computed from is never overwritten. load_synthetic
# only ever opens "synthetic.json", so each tau gets a scoring directory with
# a symlink under that name -- no copy, and no change to a loader that other
# results depend on.
#
# Generation needs the GPU; scoring does not, and runs here anyway because it
# is minutes.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=tau-sweep-fedavg
#SBATCH --time=01:30:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
SRC=_outputs/fedavg_full_hilo8_random_nes_save
RES=_outputs/results/tests
TAUS="0.8 0.9 1.0 1.1 1.2"

[ -f "$SRC/synthetic.json" ] || { echo "missing $SRC/synthetic.json" >&2; exit 1; }

# ---- generate every tau but 1.0, which is the trained model already on disk
for t in $TAUS; do
  [ "$t" = "1.0" ] && continue
  f="$SRC/synthetic_tau${t}.json"
  if [ -f "$f" ]; then echo "### skip tau=$t -- already generated"; continue; fi
  echo "### generating tau=$t  ($(date +%H:%M:%S))"
  python examples/fedpyhealth/generate.py --save-dir "$SRC" \
      --cohort-cache "$CACHE" --synth-per-hospital 8000 --temperature "$t"
done

# ---- one scoring dir per tau, symlinked so load_synthetic finds it
RUNS=""
for t in $TAUS; do
  d="_outputs/tau_fedavg_t${t}_score"
  mkdir -p "$d"
  src="synthetic.json"; [ "$t" = "1.0" ] || src="synthetic_tau${t}.json"
  [ -f "$SRC/$src" ] || { echo "missing $SRC/$src" >&2; exit 1; }
  ln -sf "$(cd "$SRC" && pwd)/$src" "$d/synthetic.json"
  RUNS="$RUNS --run tau${t}=$d"
done

mkdir -p "$RES"
export CUDA_VISIBLE_DEVICES=
# val is where tau is CHOSEN; test is where it is REPORTED. Scoring both makes
# the selection auditable instead of a number asserted in prose.
for fold in val test; do
  for scope in hospital pooled; do
    name=spec; [ "$scope" = "pooled" ] && name=gen
    out="$RES/test1_taufedavg_${fold}_${name}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo
    echo "############ tau sweep, $fold, $name  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold "$fold" --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
  done
done

echo
echo "done ($(date +%H:%M:%S))"
echo "Pick tau on test1_taufedavg_val_*, report it from test1_taufedavg_test_*."
echo "FedLIBRA to beat: +0.716 specialist / +0.704 generalist (all-codes mean)."
echo "FedAvg untuned is -0.262 / +0.552, so watch how much of that 0.98 gap"
echo "one scalar buys -- and check Pearson barely moves while R2 does."
