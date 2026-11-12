#!/bin/bash
# The five literature baselines, plus a Ditto lambda sweep.
#
# WHAT THESE ARE FOR. problem_formulation.tex claims personalized-FL methods
# "fall on a line because each of them controls one thing, how far a hospital
# may sit from the shared model". These arms are that line: Ditto and FedProx
# control the MAGNITUDE of a client's deviation, FedRep/LG-FedAvg/PerAda its
# CAPACITY. Expect them to beat our arms on the specialist axis -- local-only
# already does -- and the paper's claim is about the generalist one.
#
# HYPERPARAMETERS ARE PLACEHOLDERS. --fedprox-mu and --kd-lambda below were
# chosen to exercise the code, not read off the papers. Fix them before any of
# this is quotable.
#
# Ditto's lambda is swept because lambda->0 is exactly plain fedavg_ft, so the
# sweep anchors the line to an arm that already exists.
set -euo pipefail

SWEEP="${SWEEP:-baselines}"
MANIFEST="_outputs/results/${SWEEP}_manifest.txt"
mkdir -p _outputs/results _outputs/slurm
: > "$MANIFEST"
export SWEEP MANIFEST

sub () {   # sub <arm> <regime> <flags...>
    local arm="$1" regime="$2"; shift 2
    local id
    id=$(sbatch --parsable --job-name="bl-$arm" \
         --export=ALL,ARM="$arm",ARGS="$*",MANIFEST="$MANIFEST",REGIME="$regime" \
         examples/fedpyhealth/scripts/sbatch/_sweep_arm.sh)
    echo "  $id  $arm  ($regime)  <- $*"
    IDS="${IDS:+$IDS:}$id"
}

# Magnitude control.
sub ditto0    ditto  --ditto-lambda 0
sub ditto0.1  ditto  --ditto-lambda 0.1
sub ditto1    ditto  --ditto-lambda 1
sub ditto10   ditto  --ditto-lambda 10
sub fedprox   fedprox --fedprox-mu 0.01

# Capacity control.
sub fedrep    fedrep
sub lgfedavg  lg_fedavg
sub perada    perada --adapter lora_head --kd-lambda 1.0

score=$(sbatch --parsable --dependency=afterok:"$IDS" \
        --export=ALL,SWEEP="$SWEEP",MANIFEST="$MANIFEST" \
        examples/fedpyhealth/scripts/sbatch/_sweep_score.sh)
echo
echo "scoring job $score"
echo
echo "First minute is the real test -- the wiring has never executed. Watch:"
echo "  tail -20 _outputs/slurm/bl-fedrep-*.out      # personal_mode line"
echo "  grep 'v - w\\*' _outputs/slurm/bl-ditto10-*.out"
