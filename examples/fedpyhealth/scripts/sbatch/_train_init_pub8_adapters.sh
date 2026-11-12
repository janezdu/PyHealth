#!/bin/bash
# Adapter grid for FedAvg warm-started from pub8. Same setup as task 3 of
# _train_init_pub8.sh (lora_attn r8 lr 1e-4, which barely left zero-shot), varying
# adapter, rank and adapter learning rate. Each round logs "adapter moved", to
# separate "barely trained" from "trained to no effect".
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=init-pub8-ad
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
: "${INIT_FROM:=_outputs/centralized_full_pub8_nrounds25_localepochs2_save}"

GRID=("head_r8_lr1e-4  lora_head 8  0"
      "head_r32_lr1e-4 lora_head 32 0"
      "attn_r32_lr1e-4 lora_attn 32 0"
      "head_r8_lr1e-3  lora_head 8  1e-3"
      "attn_r8_lr1e-3  lora_attn 8  1e-3"
      "head_r32_lr1e-3 lora_head 32 1e-3"
      "attn_r32_lr1e-3 lora_attn 32 1e-3")
read -r TAG AD RANK LR <<< "${GRID[$SLURM_ARRAY_TASK_ID]}"
NAME=init_pub8_$TAG

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py --profile full \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --init-from "$INIT_FROM" --regime fedavg \
    --n-rounds 10 --local-epochs 10 --weighting uniform --metrics privacy \
    --adapter "$AD" --adapter-rank "$RANK" --adapter-lr "$LR" \
    --run-name "$NAME" --resume
echo done
