#!/bin/bash
# Fine-tune the pub8 public checkpoint on the hilo8 private sites, four ways.
#
#   0 centralized  -- pool the 8 sites' train folds, full fine-tune
#   1 local        -- each site alone, full fine-tune, never aggregated
#   2 fedavg       -- FedAvg over all 6.2M params
#   3 fedavg_lora  -- FedAvg with the pub8 trunk frozen; only a rank-8 LoRA on
#                     attention q,v (32,768 params) is trained and averaged
#
# Same budget for every arm: 10 rounds x 10 local epochs = 100 passes over each
# site's data, uniform weighting, early stopping on val (default). Matches the
# from-scratch control dp_ctrl_fedavg except for the warm start and ES.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=init-pub8
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
: "${INIT_FROM:=_outputs/centralized_full_pub8_nrounds25_localepochs2_save}"

case $SLURM_ARRAY_TASK_ID in
 0) NAME=init_pub8_centralized; ARGS="--regime centralized" ;;
 1) NAME=init_pub8_local;       ARGS="--regime local" ;;
 2) NAME=init_pub8_fedavg;      ARGS="--regime fedavg" ;;
 3) NAME=init_pub8_fedavg_lora; ARGS="--regime fedavg --adapter lora_attn" ;;
esac

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py --profile full \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --init-from "$INIT_FROM" \
    --n-rounds 10 --local-epochs 10 --weighting uniform --metrics privacy \
    $ARGS --run-name "$NAME" --resume
echo done
