#!/bin/bash
# Validate best-of-K before spending anything on it.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/smoke_xm.sbatch
#
# THE FAILURE TO CATCH. If dropout does not separate the K candidates their
# losses are equal, min() picks one arbitrarily, and the run trains on 1/K of
# its gradient -- slower, no benefit, and completely invisible in the loss
# curve. The two numbers that expose it are printed per epoch:
#
#   xm_spread       mean (max - min) of the K per-patient losses. -> 0 is dead.
#   xm_win_entropy  normalised to [0,1]; 1.0 means every candidate wins equally
#                   often. A collapse toward 0 is the same failure seen from the
#                   other side (one candidate always winning).
#
# Also checks the guard: XM and IRM together is not a defined objective, since
# IRM differentiates the risk of a FIXED pass and best-of-K makes the risk a
# minimum over stochastic ones.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-xm
#SBATCH --time=00:40:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

echo "############ 1/3  K=1 must be bit-identical to no XM at all"
python examples/fedpyhealth/train.py --profile tiny --regime centralized \
    --cohort-cache "$CACHE" --xm-k 1 --run-name smoke_xm_k1 --no-tb 2>&1 | tail -5

echo
echo "############ 2/3  K=3, the real probe"
# Filter rather than tail: the per-epoch xm lines print DURING training, so a
# plain `tail` keeps only the final eval block and silently drops the very
# numbers this smoke exists to show.
python examples/fedpyhealth/train.py --profile tiny --regime centralized \
    --cohort-cache "$CACHE" --xm-k 3 --run-name smoke_xm_k3 --no-tb 2>&1 \
    | grep -E "xm K=|Saved results|Traceback|Error" | head -30

echo
echo "---- the two numbers that matter, above:"
echo "     spread -> 0 means dropout did not separate the K candidates and"
echo "     min() is arbitrary; win_entropy -> 0 means one candidate always wins."
echo "     Either way the run trains on 1/K of its gradient and looks healthy."

echo
echo "############ 3/3  guard: XM + IRM must be refused"
if python examples/fedpyhealth/train.py --profile tiny --regime centralized \
        --cohort-cache "$CACHE" --xm-k 3 --irm-rho 100 \
        --run-name smoke_xm_guard --no-tb >/dev/null 2>&1; then
    echo "GUARD FAILED: --xm-k 3 with --irm-rho 100 was accepted" >&2
    exit 1
fi
echo "ok, rejected: --xm-k 3 with --irm-rho 100"
