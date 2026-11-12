#!/bin/bash
# The control the ouroboros arms never had: --selftrain-source local.
#
# Every self-training run so far samples the BROADCAST GLOBAL weights, so a
# small site mixes in patterns from the other seven. That is federated
# distillation, and it is the only reason to expect more than regularisation.
# Sampling the LOCAL model instead feeds a site its own distribution back --
# no information it does not already have.
#
# If global beats local, the gain is cross-site transfer. If they tie,
# self-training is a regulariser and the federation part contributes nothing.
#
# WHAT THIS CANNOT SEPARATE, and it matters for reading the result: the local
# model is trained on <=1832 patients, so its samples are worse in absolute
# terms as well as narrower in origin. A win for `global` is therefore
# "cross-site information" OR "the global model simply generates better", and
# this pair alone cannot tell them apart.
#
# At --selftrain-at 0 the two sources are identical, since nothing has trained
# yet. At 0.8 the client has finished 1 of 2 local epochs, about as far apart
# as this config gets -- which is why the pair is run there.
#
# Each arm mirrors an already-scored `global` twin at --selftrain-start 20.
set -euo pipefail

SWEEP="${SWEEP:-ourolocal}"
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

ST="--selftrain-frac 0.5 --selftrain-at 0.8 --selftrain-source local --num-synth 16000 --selftrain-start 20"
sub loc20          $ST
sub loc20_irm      $ST --irm-rho 100 --irm-warmup 17
sub loc20_xm4_irm  $ST --xm-k 4 --dropout 0.3 --irm-rho 100 --irm-warmup 17

score=$(sbatch --parsable --dependency=afterok:"$IDS" \
        --export=ALL,SWEEP="$SWEEP",MANIFEST="$MANIFEST" \
        examples/fedpyhealth/scripts/sbatch/_sweep_score.sh)
echo
echo "scoring job $score"
echo
echo "The global twins, already scored, all-codes R2 specialist/generalist:"
echo "  st20          +0.??? (test1_ouro2)      loc20         <- this run"
echo "  st20_irm      +0.346 / +0.914           loc20_irm     <- this run"
echo "  st20_xm4_irm  +0.482 / +0.951           loc20_xm4_irm <- this run"
