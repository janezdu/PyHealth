#!/bin/bash
# FedProx mu, swept down four orders of magnitude.
#
# The first attempt used mu=0.01 -- a reasonable value in Li et al., where the
# loss is O(1). Here the penalty is a SUM over 6.2M parameters and the risk is
# a MEAN near 5e-4, so 0.01 contributed roughly sixty times the risk and pinned
# every client to the broadcast weights. That arm emitted 17.7 codes per visit
# against a real 12.5 and scored R^2 = -4.7.
#
# mu=0 is the control: it must reproduce plain fedavg. If it does not, the
# penalty is wired wrong rather than merely mis-scaled.
set -euo pipefail

SWEEP="${SWEEP:-fedproxmu}"
MANIFEST="_outputs/results/${SWEEP}_manifest.txt"
mkdir -p _outputs/results _outputs/slurm
: > "$MANIFEST"
export SWEEP MANIFEST

sub () {
    local arm="$1"; shift
    local id
    id=$(sbatch --parsable --job-name="px-$arm" \
         --export=ALL,ARM="$arm",ARGS="$*",MANIFEST="$MANIFEST",REGIME=fedprox \
         examples/fedpyhealth/scripts/sbatch/_sweep_arm.sh)
    echo "  $id  $arm  <- $*"
    IDS="${IDS:+$IDS:}$id"
}

sub prox0    --fedprox-mu 0
sub prox1e6  --fedprox-mu 1e-6
sub prox1e5  --fedprox-mu 1e-5
sub prox1e4  --fedprox-mu 1e-4

score=$(sbatch --parsable --dependency=afterok:"$IDS" \
        --export=ALL,SWEEP="$SWEEP",MANIFEST="$MANIFEST" \
        examples/fedpyhealth/scripts/sbatch/_sweep_score.sh)
echo
echo "scoring job $score"
echo "watch the runtime: the reference is now built once per round, not once"
echo "per client, so these should take ~25 min rather than 2h15."
