#!/bin/bash
# RLSP for the LoRA-head-only fine-tune -- the missing control.
#
# The combos figure currently pairs two fine-tune bars that are not comparable:
#
#   lat_lora_ft   LoRA head (37,792 params), 20 epochs, + IRM + best-of-K
#   fedavg_ft     FULL model (6,198,104 params), 2 epochs, nothing else
#
# So the gap between them mixes "which parameters move", "for how long", and
# "which mechanisms are on". lora_head20 is LoRA head / 20 epochs / no IRM /
# no best-of-K, which isolates the adapter: against fedavg_ft it says what
# constraining the update buys, and against lat_lora_ft it says what the
# mechanisms add on top of the adapter.
#
# Natural mix, rnn, seeds 0/1/2, settings copied verbatim from
# _test2_rlsp_all.sh so this file merges onto the same axis as the 14 arms
# already scored there. Regeneration first: this run holds 5000 synthetic per
# hospital and --train-budget 8000 needs 8000.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-lorahead
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
D=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_save
[ -d "$D" ] || { echo "missing $D" >&2; exit 1; }

echo "############ regenerating lora_head20 at 8000/hospital"
python examples/fedpyhealth/generate.py --save-dir "$D" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000

for seed in 0 1 2; do
    out="_outputs/results/tests/test2_rlsplora_natural_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ rlsp lora_head20, natural, seed $seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" \
        --run lora_head20="$D" \
        --rlsp --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
