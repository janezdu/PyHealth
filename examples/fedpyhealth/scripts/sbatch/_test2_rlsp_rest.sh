#!/bin/bash
# RLSP for every remaining arm on the frontier figure.
#
# RLSP coverage before this job was 7 generators, and only in ONE cell
# (rnn / weight 0.5 / upsample): fedavg, fedavg_irm, fedavg_irm_xm,
# fed_xm4_do03, xm4, st20, st20_xm4_irm. Every other cell had just three.
# These six were scored on TSTR only, which asks whether synthetic REPLACES a
# site's real records -- not the question the figure is about.
#
#   fedavg_irm1e4  IRM rho=1e4, no best-of-K        shared generator
#   lat_fed_irm    latent z=16 + IRM + best-of-K    shared generator
#   fedavg_ft      the 2-epoch full fine-tune       per hospital
#   lat_lora_ft    latent + IRM + best-of-K + FT    per hospital
#   ditto0.1       Ditto lambda=0.1                 per hospital
#   perada         PerAda (LoRA head + KD)          per hospital
#
# Settings copied verbatim from _test2_rlsp_irm4arm.sh -- rnn, 50/50 upsample,
# 30 drawn codes, min 20 positives, budget 8000, seeds 0/1/2 -- so this file
# merges onto the same axis as the seven already scored. The 30 eval codes are
# drawn deterministically from the fold's scoreable set given --eval-seed, so
# seed N here draws the same codes as seed N there.
#
# KNOWN AND ACCEPTED: the four per-hospital arms contribute 8 x 8000 synthetic
# records to the pooled half, against 8000 for a shared generator, because RLSP
# unions every generator in an arm. Flagged, and explicitly not corrected for.
#
# WALL CLOCK. t2-rlsp-irm4 (5 shared arms, 3 seeds) ran 23m43s; t2-rlsp-ouro
# (2 arms) 15m44s. This job trains ~82 classifiers per seed against irm4's ~45,
# because a per-hospital arm needs 8 tstr plus 8 rlsp classifiers rather than
# 1 plus 8. Call it ~45min; 03:00:00 is the margin, not the estimate.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-rest
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run fedavg_irm1e4=_outputs/fedavg_full_hilo8_random_irmrho10000_save
      --run lat_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_z16_xm4_save
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_nes_save
      --run lat_lora_ft=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save
      --run ditto0.1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save
      --run perada=_outputs/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for seed in 0 1 2; do
    out="$RES/test2_rlsp_rest_upsample_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ rlsp rest, seed $seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --rlsp-weight 0.5 --rlsp-weight-mode upsample --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
