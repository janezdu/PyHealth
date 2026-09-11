#!/bin/bash
# Two fedavg sweeps, submitted as independent GPU jobs plus one dependent
# scoring job. Run from the repo root:  bash examples/fedpyhealth/scripts/sweep_fedavg.sh
#
# SWEEP 1 -- dropout, higher. Federated best-of-K exists at dropout 0.1 and 0.3
# and is flat at both (~0.54 head AUROC, against a centralized K=1 control at
# 0.582). Dropout is what SUPPLIES the candidate diversity K selects over, so if
# federated XM is failing for want of spread, more dropout is the direct fix --
# and the training logs support that reading: federated spread ran ~30% below
# centralized at the same K. 0.5 and 0.7 are well past anything tried here, so
# treat a collapse in sample quality as an expected outcome, not a bug.
#
# Also included: K=1 at dropout 0.3, the control federation has never had.
#
# SWEEP 2 -- latent width. Only z=16 exists. Note this arm runs DROPOUT 0.0, so
# the latent is the ONLY source of candidate diversity for its K=4 best-of-K;
# z is therefore the exploration knob here, not just model capacity. z=4 tests
# whether that diversity is too narrow to matter, z=32/64 whether it is too
# wide to average cleanly under FedAvg.
#
# Everything else is held at the values the existing arms used, so each new run
# differs from a scored one in exactly one flag.
set -euo pipefail

SWEEP="${SWEEP:-fedsweep}"
MANIFEST="_outputs/results/${SWEEP}_manifest.txt"
mkdir -p _outputs/results _outputs/slurm
: > "$MANIFEST"          # fresh -- a stale manifest would score last run's arms
export SWEEP MANIFEST

sub () {   # sub <arm-label> <train.py flags...>
    local arm="$1"; shift
    local id
    id=$(sbatch --parsable --job-name="sw-$arm" \
         --export=ALL,ARM="$arm",ARGS="$*",MANIFEST="$MANIFEST" \
         examples/fedpyhealth/scripts/sbatch/_sweep_arm.sh)
    echo "  $id  $arm  <- $*"
    IDS="${IDS:+$IDS:}$id"
}

echo "SWEEP 1: dropout, higher (0.1 and 0.3 already scored at K=4)"
sub do05 --dropout 0.5 --xm-k 4
sub do07 --dropout 0.7 --xm-k 4
# The control the federated K ladder has never had. One job, and without it
# "K does nothing federated" stays an inference rather than a measurement.
sub k1   --dropout 0.3 --xm-k 1

echo "SWEEP 2: latent width, IRM rho=100 warmup 17, dropout 0 (z=16 already scored)"
for z in 4 32 64; do
    sub "z$z" --irm-rho 100 --irm-warmup 17 --latent-dim "$z" --xm-k 4
done

score=$(sbatch --parsable --dependency=afterok:"$IDS" \
        --export=ALL,SWEEP="$SWEEP",MANIFEST="$MANIFEST" \
        examples/fedpyhealth/scripts/sbatch/_sweep_score.sh)
echo
echo "scoring job $score (waits on all five)"
echo "results -> _outputs/results/tests/test1_${SWEEP}_{spec,gen}_b200.json"
echo "           _outputs/results/tests/test2_head_${SWEEP}_seed{0,1,2}.json"
echo
echo "CHECK THE FIRST EPOCH of each K arm:"
echo "  grep -m2 'xm K=' _outputs/slurm/sw-k16-*.out"
echo "spread=0.000 means the K candidates are identical and the run is inert."
