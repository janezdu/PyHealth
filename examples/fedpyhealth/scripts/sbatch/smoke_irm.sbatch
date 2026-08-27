#!/bin/bash
# Five-minute validation of the IRM wiring, before spending real GPU hours.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/smoke_irm.sbatch
#
# The IRM penalty touches shared library code (HALOModel.forward,
# HALO._irm_terms, HALO.train_model) and all four training helpers. Two of those
# paths -- fedavg_ft's fine-tuning and local's per-site resume -- are not reached
# until well into a real run, so a typo there surfaces hours in. This exercises
# every path on the tiny profile first.
#
# It also runs the unit tests here rather than on the login node: importing
# torch there takes minutes and the cluster rules say not to.
#
# WHAT WOULD MAKE THIS FAIL LOUDLY, and should:
#   * a penalty with no gradient path to theta (missing create_graph) -- caught
#     by test_penalty_is_differentiable_wrt_parameters
#   * irm_scale changing the loss when it should be inert -- caught by
#     test_none_matches_scale_one
#   * a batch too small to split at the smallest hospital -- _irm_terms raises
#     with the hospital's batch size in the message
#
# WHAT IT CANNOT TELL YOU: whether IRM helps. The tiny profile is 2 layers for a
# handful of epochs; the penalty will be noise. This checks that the code runs,
# nothing more.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-irm
#SBATCH --time=00:40:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

CACHE="$FEDCOHORT_CACHE/hilo8_random"

echo "############ 1/3  unit tests"
python -m unittest tests.core.test_halo_irm tests.core.test_halo_sample_weights -v

echo
echo "############ 2/3  tiny run, IRM OFF -- the control"
# If this differs from a pre-IRM tiny run, the penalty is not inert when off and
# every existing checkpoint has been invalidated.
python examples/fedpyhealth/train.py --profile tiny --regime fedavg \
    --cohort-cache "$CACHE" --run-name smoke_irm_off --no-tb

echo
echo "############ 3/3  tiny runs, IRM ON, every regime"
for regime in fedavg fedavg_ft centralized local; do
    echo "---- $regime"
    python examples/fedpyhealth/train.py --profile tiny --regime "$regime" \
        --cohort-cache "$CACHE" --irm-rho 100 --irm-warmup 2 \
        --run-name "smoke_irm_${regime}" --no-tb
done

echo
echo "All four regimes ran with the penalty on. Check the logs above for the"
echo "risk/penalty split before launching the full runs."
