#!/bin/bash
# Score the 18 b128 replicate runs -- the Figure 1 confidence intervals.
#
# WHAT IS ON DISK. 6 arms x 3 independent runs (reps 2,3,4), every one at
# batch_size 128 with the PUBLISHED per-arm treatment (centralized alone gets
# upweighting + early stopping). Training and generation are done; nothing has
# been scored. This is the one matched setting in which all six Figure 1 points
# exist, which is what the whole b128 campaign was for.
#
# THESE ARE NOT SEEDS. train.py hardcodes SEED = 0 and has no --seed flag, so a
# rerun under a new --run-name is an independent draw only because GPU training
# is nondeterministic. Call them replicate runs in the paper, never seeds.
#
# AGGREGATE AS THE MAIN TABLE DOES: mean over the 8 hospitals of
# PrevVal_All_Prevalence_R2, then mean and SD over the three runs. Median and
# the rare-code columns give different answers -- the four statistics disagree
# by more than most arm-to-arm gaps, so mixing them silently is how the figure
# got inconsistent in the first place.
#
# 200 bootstraps. At the default 5 the spread is pure quantisation noise and
# swamps every difference worth reporting.
#
# CPU ONLY -- prevalence scoring touches no model.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=score-b128reps
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS=""
for rep in 2 3 4; do
  for arm in local centralized fedavg ft_full ft_lora fedlibra; do
    d="_outputs/b128rep${rep}_${arm}_save"
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
    RUNS="$RUNS --run ${arm}_rep${rep}=$d"
  done
done

mkdir -p "$RES"
for scope in hospital pooled; do
  name=spec; [ "$scope" = "pooled" ] && name=gen
  out="$RES/test1_b128reps_${name}_b200.json"
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo
  echo "############ b128 reps, $name  ($(date +%H:%M:%S))"
  python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
      --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
      --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done

echo
echo "done ($(date +%H:%M:%S))"
echo "Plot with examples/fedpyhealth/scripts/fig_frontier_uniform.py pointed at"
echo "test1_b128reps_{spec,gen}_b200.json."
