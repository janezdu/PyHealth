#!/bin/bash
# Validate the delta-sparse adapters before spending GPU hours on them.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/smoke_sparse_adapters.sbatch
#
# These two variants train exactly the weights last_mlp trains and differ only
# in what the UPDATE is allowed to be: sparse in coordinates, where LoRA is
# low-rank in directions. Sparsity is measured on D = W - W_global, so a site
# changes a few coordinates and is the federated global everywhere else.
#
# Everything worth checking here fails silently, and in both directions:
#
#   * a subgradient L1 shrinks coordinates but never lands on zero, so the run
#     regularises, reports 0% sparsity, and reads as a failed idea rather than a
#     failed representation. The proximal step is what makes actual zeros;
#   * hard thresholding that does not also mask Adam's exp_avg has the very next
#     step push every pruned coordinate back off zero -- the run ends dense while
#     each projection looked like it worked;
#   * sparsity is invisible in the training loss. A dense run and a 7%-dense one
#     sit at indistinguishable losses, so the loss curve cannot catch either bug.
#     ||D||_1 and the non-zero fraction are logged per epoch for exactly this.
#
# tests/core/test_halo_adapters.py covers all three. This runs them on a compute
# node (importing torch on the login node takes minutes) and then exercises each
# variant end to end, because the unit tests bind the helpers directly and
# cannot catch a break in how train.py calls them.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-sparse
#SBATCH --time=00:40:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

CACHE="$FEDCOHORT_CACHE/hilo8_random"

echo "############ 1/3  unit tests (all adapters, sparse included)"
python -m unittest tests.core.test_halo_adapters -v

echo
echo "############ 2/3  tiny end-to-end, the two sparse variants"
echo "---- last_mlp_iht, 50% density, projecting every epoch"
python examples/fedpyhealth/train.py --profile tiny --regime fedavg_ft \
    --cohort-cache "$CACHE" --adapter last_mlp_iht --adapter-sparsity 0.5 \
    --adapter-iht-every 1 --run-name smoke_sparse_iht --no-tb

echo "---- last_mlp_l1, proximal"
python examples/fedpyhealth/train.py --profile tiny --regime fedavg_ft \
    --cohort-cache "$CACHE" --adapter last_mlp_l1 --adapter-l1 0.1 \
    --run-name smoke_sparse_l1 --no-tb

echo
echo "############ 3/3  the SGD cross-check arm"
# lr*lambda is the EXACT proximal operator under SGD (ISTA) and only a heuristic
# under Adam, and stateless SGD cannot have a pruned coordinate re-inflated by a
# momentum buffer. It needs its own lr: 1e-4 is an ADAM learning rate and SGD
# barely moves at it, so an untuned SGD arm underperforms for reasons that have
# nothing to do with sparsity.
python examples/fedpyhealth/train.py --profile tiny --regime fedavg_ft \
    --cohort-cache "$CACHE" --adapter last_mlp_iht --adapter-sparsity 0.5 \
    --adapter-optim sgd --adapter-lr 0.01 \
    --run-name smoke_sparse_sgd --no-tb

echo
echo "############ guards: these MUST fail"
for bad in "--adapter last_mlp --adapter-l1 0.1" \
           "--adapter last_mlp_iht" \
           "--adapter last_mlp_l1"; do
    if python examples/fedpyhealth/train.py --profile tiny --regime fedavg_ft \
            --cohort-cache "$CACHE" $bad --run-name smoke_guard --no-tb \
            >/dev/null 2>&1; then
        echo "GUARD FAILED: '$bad' was accepted and should not have been" >&2
        exit 1
    fi
    echo "ok, rejected: $bad"
done

echo
echo "All variants ran. In the logs above check that:"
echo "  * every hospital line reads 'training 33,088 of 2,115,288 params' --"
echo "    that is the TINY architecture; the full profile's last MLP is 525,568"
echo "  * the iht runs report a density near 0.5, not near 1.0"
echo "  * the l1 run's density is BELOW 1.0 -- if it is exactly 1.0 the"
echo "    proximal step is not firing and the run is plain last_mlp"
