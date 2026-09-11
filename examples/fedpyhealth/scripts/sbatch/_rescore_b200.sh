#!/bin/bash
# Re-score every Test 1 result at 200 bootstrap resamples with a fixed seed.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/_rescore_b200.sh
#
# NO REGENERATION. Every arm's synthetic.json is already on disk at
# 8000/hospital; this only re-runs the metric over it. That is the whole point:
# the imprecision was never in the generator.
#
# WHY. compute_prevalence_metrics resamples over CODES and reports the mean of
# n_bootstraps draws. The default is 5. Measured on this cohort the per-run
# bootstrap std on specialist R^2 is ~0.22, so a 5-resample mean carries a
# standard error near 0.10 -- and three scorings of three BIT-IDENTICAL models
# (the degenerate lambda=1/10/100 arms) spanned 0.184, which that alone
# explains. Every specialist-R^2 comparison made this session sits inside it.
#
#   resamples   specialist R^2 SE
#          5    0.098   <- what every result to date used
#         50    0.031
#        200    0.016
#
# Seeding is separate and does NOT fix this: it makes one arbitrary draw
# reproducible. --bootstrap-seed 0 is set so a re-run of THIS job reproduces,
# and it is shared across arms so they are resampled identically.
#
# Writes to *_b200.json. Nothing existing is overwritten -- the old numbers stay
# on disk so the before/after is inspectable.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=rescore-b200
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests
B=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapter
SP=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_nes_last_mlp

echo "############ 1/3  the new seeding unit tests"
# Scoped to TestBootstrapSeed rather than the whole module on purpose. The
# module also contains test_code_subset_keeps_the_full_patient_denominator,
# which fails on its OWN guard ("fixture too dense to exercise the bug"): the
# 5-patient fixture is fully covered by its first two codes, so assertLess(5, 5)
# can never hold and the test cannot exercise what it was written to check.
# That is a pre-existing repo bug, unrelated to anything here, and gating a
# rescore on it would block work for a reason that has nothing to do with the
# rescore.
python -m unittest \
    tests.core.test_generative_metrics.TestBootstrapSeed -v 2>&1 | tail -40

score () {   # score <stem> <run-args...>
    local stem="$1"; shift
    echo "############ $stem  (specialist)"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" "$@" \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "$RES/${stem}_spec_b200.json"
    echo "############ $stem  (generalist)"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" "$@" \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "$RES/${stem}_gen_b200.json"
}

echo "############ 2/3  the published plot: ft20 family + IRM trunk"
score test1_adapters20 \
    --run full_ft20=${B}none_save \
    --run lora_attn20=${B}lora_attn_save \
    --run lora_head20=${B}lora_head_save \
    --run last_mlp20=${B}last_mlp_save \
    --run irm_full_ft=${B}none_irmrho100_irmwarmup34_save \
    --run irm_lora_head=${B}lora_head_irmrho100_irmwarmup34_save \
    --run irm_lora_attn=${B}lora_attn_irmrho100_irmwarmup34_save

echo "############ 3/3  the sparse ladder + the recalibrated lambdas"
score test1_sparse \
    --run iht50=${SP}_ihtr8_sp0.5_save \
    --run iht10=${SP}_ihtr8_sp0.1_save \
    --run iht07=${SP}_ihtr8_sp0.07_save \
    --run iht10_sgd=${SP}_ihtr8_sp0.1_sgd_alr1_save \
    --run l1_02=${SP}_l1r8_l10.2_save \
    --run l1_03=${SP}_l1r8_l10.3_save \
    --run l1_05=${SP}_l1r8_l10.5_save \
    --run zero_a=${SP}_l1r8_l11_save \
    --run zero_b=${SP}_l1r8_l110_save \
    --run zero_c=${SP}_l1r8_l1100_save

echo
echo "Done. zero_a/b/c are the three degenerate lambda arms -- bit-identical"
echo "models, so their remaining spread at 200 resamples is the noise floor"
echo "that every other comparison must clear."
