#!/bin/bash
# RLSP for the two OUROBOROS arms, to sit beside the four-arm fedavg figure.
#
# WHY A SEPARATE FILE AND NOT A RERUN OF _test2_rlsp_irm4arm.sh. The ouroboros
# arms were Test-2 scored only as TSTR (synthetic REPLACES the site's real
# records), which answers a different question than the RLSP figure asks. The
# four arms already in test2_rlsp_irm4arm_upsample_rnn_seed*.json do not need
# rescoring -- every setting below is copied from that script verbatim, and the
# 30 eval codes are drawn deterministically from the fold's 91 scoreable codes
# given --eval-seed, so a seed-N file here draws the SAME codes as a seed-N file
# there. Scoring only the new arms costs ~half the GPU hours of a full rerun.
#
#   st20          FedAvg + ouroboros (self-train 0.5 from round 20, source global)
#   st20_xm4_irm  the same, plus IRM rho=1e2 warmup 17 AND best-of-K K=4, do 0.3
#
# THE CONFOUND CARRIES OVER, and gains one. As in the four-arm job, the arms do
# not share dropout or batch size (st20 is do 0 / batch 128; st20_xm4_irm is
# do 0.3 / batch 256). On top of that, st20_xm4_irm changes THREE things at once
# against st20 -- IRM, best-of-K, and dropout -- so the gap between these two
# bars is not attributable to any one of them. It is a "best combined recipe"
# bar, not an ablation. Read it that way or not at all.
#
# real_local is recomputed here rather than reused from the four-arm file. It
# costs 8 of the 24 classifiers but keeps this file self-contained, and the two
# estimates agreeing is a free check that the seeds line up.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-ouro
#SBATCH --time=04:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run st20=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_st0.5r20@1_save
      --run st20_xm4_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.3_xm4_st0.5r20@1_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for seed in 0 1 2; do
    out="$RES/test2_rlsp_ouro_upsample_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ rlsp ouroboros, seed $seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --rlsp-weight 0.5 --rlsp-weight-mode upsample --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
