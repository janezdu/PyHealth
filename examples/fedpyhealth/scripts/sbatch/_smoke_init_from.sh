#!/bin/bash
# Smoke test for --init-from and the whole-run frozen trunk (--adapter outside
# the FT regimes). One round, one local epoch, small synthetic sets: this
# checks that each arm warm-starts and runs end to end, not what it scores.
#
# Read each log for:
#   "Init: warm start from ... vocabulary matches (921 codes)"
#   first-epoch loss ~0.0004-0.001 (pub8 from random weights opened at 0.0137;
#     a number near that means the warm start did not take)
#   task 3 only: "trunk frozen -- each client sends ~32,768 of ~6.2M params"
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-init
#SBATCH --time=00:45:00
#SBATCH --array=0-3
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
: "${INIT_FROM:=_outputs/centralized_full_pub8_nrounds25_localepochs2_save}"

case $SLURM_ARRAY_TASK_ID in
 0) NAME=smoke_init_centralized; ARGS="--regime centralized" ;;
 1) NAME=smoke_init_local;       ARGS="--regime local" ;;
 2) NAME=smoke_init_fedavg;      ARGS="--regime fedavg" ;;
 3) NAME=smoke_init_fedavg_lora; ARGS="--regime fedavg --adapter lora_attn" ;;
esac

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py --profile full \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --init-from "$INIT_FROM" \
    --n-rounds 1 --local-epochs 1 --num-synth 200 --synth-per-hospital 50 \
    $ARGS --run-name "$NAME"
echo done
