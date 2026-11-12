#!/bin/bash
# THE ROW-COUNT CONTROL FOR RLSP. One question: is the RLSP utility gain the
# synthetic patients, or is it just 40x more training rows?
#
# An rlsp:<regime>:<hid> arm differs from real_local:<hid> in TWO ways -- it
# adds synthetic patients AND it goes from 142-1,832 rows to ~6,600. The
# appendix already states this RNN overfits a site's real data alone (which is
# why real_local lands at 0.420, below chance), and row count alone relieves
# overfitting. So the two explanations are confounded in every number we
# report.
#
# --rlsp-dup-control adds rlsp_dup:<regime>:<hid>: that site's OWN records
# duplicated to the SAME row count as its mixture, zero synthetic. Same rows,
# same optimizer steps, no new information. Read the ladder:
#
#   real_local  ->  rlsp_dup     what more ROWS buy
#   rlsp_dup    ->  rlsp         what SYNTHETIC PATIENTS buy
#
# If rlsp_dup lands at rlsp, the synthetic half was padding. If rlsp clears it,
# the generated patients carry information the site did not have -- a stronger
# claim than the current table makes, and one that survives the obvious
# objection.
#
# NOTHING ELSE CHANGES. Same RNN (128-dim, 10 epochs, no early stopping), same
# cohort, same code draws, same budget as the published numbers. No classifier
# swap, no architecture sweep. Only the baselines in this file are rescored;
# every other arm in the paper keeps its published number, because no other arm
# has a pathological row count.
#
# Two regimes: lat_lora_ft is FedLIBRA, fedavg is the reference it is read
# against. Natural mix only -- the 50/50 cells are a separate question.
#
# WALL CLOCK. ~60 classifiers per seed at ~10.6s each -> ~11m/seed, ~35m total.
# 02:00:00 is margin.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-dup
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run lat_lora_ft=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for seed in 0 1 2; do
    out="$RES/test2_rlsp_dupctrl_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ rlsp dup control, seed=$seed"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --rlsp-dup-control --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
