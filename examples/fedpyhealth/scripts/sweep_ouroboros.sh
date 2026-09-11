#!/bin/bash
# Ouroboros with a WARMUP. Run from the repo root:
#   bash examples/fedpyhealth/scripts/sweep_ouroboros.sh
#
# WHY A WARMUP. The v1 run (ouroboros_v1-000-global, --selftrain-start 0) mixed
# synthetic in from round 1, when the global model emitted 460 codes/visit
# against a real cohort mean of 12.53 -- essentially the entire 921-code
# vocabulary in every patient. Half of every client's training data was that.
# The run then oscillated (460 -> 50 -> 459 -> 8.2) and ended BELOW real with
# distinct codes falling 919 -> 811, which is mode collapse, not convergence.
# Nothing in that tests self-training; it tests whether a model survives being
# fed noise. Starting later is the fix.
#
# WHAT CHANGED FROM v1, three things at once, so read this as a new baseline
# rather than a controlled delta against it:
#   1. --selftrain-start, swept: 0 (the v1 control), 10, 20, 30 of 50 rounds.
#   2. The E2_R50 profile every plotted arm uses, so results are comparable.
#      v1 ran 10 rounds x 10 local epochs and is comparable to nothing.
#   3. --num-synth 16000. v1 used 5000, which resolves prevalence to 0.0002
#      while the rarest scored code sits at 0.00076 -- its rare metrics were
#      largely quantization noise. The log itself warned about this.
#
# WATCH THE VAL CURVE, which this project has never had: train.py now logs
# loss_val/global every round even with --no-early-stop. Self-training fails by
# the model drifting onto its own output, and that shows up as val loss rising
# while train loss keeps falling -- long before the prevalence metrics land.
#
#   grep 'val ' _outputs/slurm/sw-st*-*.out
#
# Also watch codes/visit per round; it should stay near 12.53, not tour 8-460.
set -euo pipefail

SWEEP="${SWEEP:-ouro2}"
MANIFEST="_outputs/results/${SWEEP}_manifest.txt"
mkdir -p _outputs/results _outputs/slurm
: > "$MANIFEST"
export SWEEP MANIFEST

sub () {
    local arm="$1"; shift
    local id
    id=$(sbatch --parsable --job-name="sw-$arm" \
         --export=ALL,ARM="$arm",ARGS="$*",MANIFEST="$MANIFEST" \
         examples/fedpyhealth/scripts/sbatch/_sweep_arm.sh)
    echo "  $id  $arm  <- $*"
    IDS="${IDS:+$IDS:}$id"
}

COMMON="--selftrain-frac 0.5 --selftrain-at 0.8 --selftrain-source global --num-synth 16000"
echo "ouroboros warmup sweep (50 rounds; mixing starts at round N)"
for st in 0 10 20 30; do
    sub "st$st" $COMMON --selftrain-start "$st"
done

score=$(sbatch --parsable --dependency=afterok:"$IDS" \
        --export=ALL,SWEEP="$SWEEP",MANIFEST="$MANIFEST" \
        examples/fedpyhealth/scripts/sbatch/_sweep_score.sh)
echo
echo "scoring job $score (waits on all four)"
echo "results -> _outputs/results/tests/test1_${SWEEP}_{spec,gen}_b200.json"
echo "           _outputs/results/tests/test2_head_${SWEEP}_seed{0,1,2}.json"
echo
echo "st0 is the v1 control at the new profile -- the comparison that isolates"
echo "the warmup from the profile change. Do not compare any of these to v1."
