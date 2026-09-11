#!/bin/bash
# Does --n-bootstraps actually change our CONCLUSIONS?
#
# Run on an interactive node (no GPU needed -- test1_prevalence.py is pandas and
# sklearn, it never touches CUDA):
#
#   srun --account=bgyw-delta-gpu --partition=gpuA40x4 --gpus-per-node=1 \
#        --cpus-per-task=8 --mem=32g --time=01:00:00 --pty bash
#   source .venv/bin/activate
#   bash examples/fedpyhealth/scripts/bootstrap_sensitivity.sh
#
# WHAT IT ASKS. The reported Pearson/R^2 is the MEAN of n_bootstraps resamples
# over codes. Raising n makes that mean more stable -- but stability of the
# NUMBER is not the same as stability of the ANSWER. What actually matters is
# whether the ORDERING of arms survives, because every claim in this project is
# a comparison, not an absolute level.
#
# So this scores the same four arms at n=5 under five different seeds, and once
# at n=200, then reports:
#   * how far the n=5 estimate wanders across seeds, and
#   * whether the arm ranking changes with the seed.
#
# If the ranking is stable, n=5 is fine for comparisons and rescoring
# everything at 200 is wasted work. If it flips, it is not.
set -euo pipefail

CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"
OUT=_outputs/results/bootstrap_sensitivity
mkdir -p "$OUT"

B=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapter
SP=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_nes_last_mlp

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run lora_head20=${B}lora_head_save
      --run irm_lora_head=${B}lora_head_irmrho100_irmwarmup34_save
      --run iht07=${SP}_ihtr8_sp0.07_save
      --run l1_05=${SP}_l1r8_l10.5_save"

for seed in 0 1 2 3 4; do
    echo "#### n=5, seed $seed"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
        --n-bootstraps 5 --bootstrap-seed "$seed" \
        --out "$OUT/n5_seed${seed}.json" >/dev/null
done

echo "#### n=200, seed 0"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out "$OUT/n200_seed0.json" >/dev/null

echo "#### n=200, seed 1  (to show the n=200 estimate is itself stable)"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --n-bootstraps 200 --bootstrap-seed 1 \
    --out "$OUT/n200_seed1.json" >/dev/null

echo
python examples/fedpyhealth/scripts/bootstrap_sensitivity.py "$OUT"
