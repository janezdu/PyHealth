#!/bin/bash
# Ouroboros x best-of-K x IRM. Run from the repo root.
#
# WHY. The two families have never been combined: every self-training run so far
# has xm_k=1, irm_rho=0, and every XM/IRM run has selftrain_frac=0.
#
# THE PREDICTION, which is what makes this a test rather than a fishing trip:
# these two should COMPOUND BADLY. Ouroboros v2 converged to 4-5 codes/visit
# against a real 12.53, with distinct codes settling near 500 of 921 -- the
# model narrowing onto its own output. Best-of-K takes the minimum-loss
# candidate of K, which is also a narrowing operator: it selects the most
# typical draw. Two narrowing operators stacked should collapse harder, and
# codes/visit per round measures exactly that. If it does not, the mechanism
# I have been attributing XM's behaviour to is wrong.
#
# IRM is here with no strong hypothesis. Ouroboros collapsed near-identically
# at all eight sites, so cross-site invariance has little to act on; it is
# included because it is the only regulariser in this project that has moved
# Test 1 at all (fedavg 0.7498 -> 0.7573).
#
# --dropout 0.3 on the XM arms is not optional: HALO's forward pass is
# deterministic without it, so the K candidates come back bit-identical, min()
# picks arbitrarily, and the run silently trains on 1/K of its gradient.
# CHECK THE FIRST EPOCH for `xm K=4 spread=...`; 0.000 means exactly that.
#
# st20 from the finished ouroboros sweep is the control -- same warmup, same
# frac, K=1, no IRM -- and is already scored.
set -euo pipefail

SWEEP="${SWEEP:-ouroxm}"
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

ST="--selftrain-frac 0.5 --selftrain-at 0.8 --selftrain-source global --num-synth 16000 --selftrain-start 20"
sub st20_xm4     $ST --xm-k 4 --dropout 0.3
sub st20_irm     $ST --irm-rho 100 --irm-warmup 17
sub st20_xm4_irm $ST --xm-k 4 --dropout 0.3 --irm-rho 100 --irm-warmup 17

score=$(sbatch --parsable --dependency=afterok:"$IDS" \
        --export=ALL,SWEEP="$SWEEP",MANIFEST="$MANIFEST" \
        examples/fedpyhealth/scripts/sbatch/_sweep_score.sh)
echo
echo "scoring job $score"
echo "watch:  grep -h 'client 458 @' _outputs/slurm/sw-st20_xm4-*.out | tail -5"
echo "st20 control converged to 4.2 codes/visit; real is 12.53."
