#!/bin/bash
# RLSP utility for the two fine-tune ablation rows that have none: FedAvg +
# LoRA-head fine-tune and FedAvg + full fine-tune (both 20 epochs, the same
# fine-tune stage as FedLIBRA). Every other ablation row already has RLSP
# numbers from _test2_rlsp_all.sh.
#
# Same flags as the utility table's cells (random 30 codes from those with
# >= 20 test positives, natural mix, RNN, seeds 0-2), so the numbers drop into
# the ablation table next to the existing ones. Both runs already hold 8000
# synthetic per hospital -- no regeneration.
#
# Two arms x (8 rlsp + real_local/pooled controls) at ~10.6 s per classifier is
# ~5 min per seed; 3 seeds serial fits well inside the hour.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-abl-ft
#SBATCH --time=01:30:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run lora_head20=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_save
      --run full_ft20=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapternone_save"
for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for seed in 0 1 2; do
    out="$RES/test2_rlspablft_rand_natural_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ seed $seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp --classifier rnn --fold test --min-positives 20 \
        --code-pool all --eval-code-order random --n-eval-codes 30 \
        --eval-seed "$seed" --train-budget 8000 --out "$out"
done
echo done
