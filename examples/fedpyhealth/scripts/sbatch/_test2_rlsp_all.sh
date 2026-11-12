#!/bin/bash
# RLSP for every arm, in BOTH mixes, so no comparison is cell-locked.
#
# WHY THIS REPLACES _test2_rlsp_rest.sh. RLSP coverage was lopsided: the IRM
# and ouroboros arms existed ONLY at weight 0.5 / upsample, and the three
# original arms existed in every cell. That matters because the arm ordering
# FLIPS with the mix and not with the classifier:
#
#   rnn natural          fedavg 0.617  <  best-of-K 0.629
#   rnn 50/50 upsample   fedavg 0.678  >  best-of-K 0.661
#   rnn 50/50 fixed      fedavg 0.665  >  best-of-K 0.658
#
# Both natural cells favour best-of-K; all four 50/50 cells favour FedAvg. With
# the IRM arms present in one cell only, any statement about them was a
# statement about that cell. Every gap is 0.006-0.018 against a seed spread of
# 0.015-0.040, so the point is not which wins -- it is that the figure must be
# able to show both.
#
# Natural mix is plain --rlsp with no --rlsp-weight: 80-98% of each mixed arm's
# records are synthetic. At 50/50 the real half is duplicated up to match, so
# the generator's contribution is diluted. Those are different questions and
# both belong on the page.
#
# 14 arms. lat_xm4 is new (job 22187837, finished 19m52s): fedavg + latent
# z=16 + best-of-K, NO IRM -- the control that makes the latent best-of-K
# comparison an ablation rather than a recipe.
#
# Per-hospital arms contribute 8 x 8000 synthetic against a shared generator's
# 8000. Known, and deliberately not corrected for.
#
# WALL CLOCK. ~162 classifiers per (seed, mix) against t2-rlsp-irm4's ~45 in
# 23m43s, so ~10.6s each -> ~2.9h for 3 seeds x 2 mixes. 06:00:00 is margin.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-all
#SBATCH --time=06:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_irm=_outputs/fedavg_full_hilo8_random_irmrho100_save
      --run fedavg_irm1e4=_outputs/fedavg_full_hilo8_random_irmrho10000_save
      --run fed_xm4_do03=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run fedavg_irm_xm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.1_xm4_save
      --run lat_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_z16_xm4_save
      --run lat_xm4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_z16_xm4_save
      --run st20=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_st0.5r20@1_save
      --run st20_xm4_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.3_xm4_st0.5r20@1_save
      --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_nes_save
      --run lat_lora_ft=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save
      --run ditto0.1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save
      --run perada=_outputs/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for mix in natural upsample; do
  for seed in 0 1 2; do
    out="$RES/test2_rlspall_${mix}_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    # natural is plain --rlsp; 50/50 adds the weight flags. Nothing else differs.
    W=""; [ "$mix" = "upsample" ] && W="--rlsp-weight 0.5 --rlsp-weight-mode upsample"
    echo "############ rlsp all, mix=$mix seed=$seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp $W --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
  done
done
echo done
