#!/bin/bash
# LoRA-head fine-tuning on top of the LATENT + IRM trunk.
#
# WHY IT SEEDS FROM THE LATENT RUN, NOT THE EXISTING IRM TRUNK. A latent model
# carries z_proj parameters that a non-latent checkpoint does not have, and
# run_fedavg loads the global with strict=True -- seeding
# fedavg_full_hilo8_random_irmrho100_save would fail on missing keys. So the
# trunk has to come from the latent+IRM run this job depends on.
#
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=lat-lora-ft
#SBATCH --time=04:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
COMMON="--profile full --cohort-cache $CACHE --no-early-stop --latent-dim 16 \
        --xm-k 4 --irm-rho 100 --irm-warmup 17"

# Ask train.py for both paths rather than reconstructing them -- guessing the
# component order has already cost two runs this week.
TRUNK_DIR=$(python examples/fedpyhealth/train.py $COMMON --regime fedavg --print-save-dir | tail -1)
DEST=$(python examples/fedpyhealth/train.py $COMMON --regime fedavg_ft --ft-epochs 20 \
        --adapter lora_head --resume --print-save-dir | tail -1)
echo "trunk: $TRUNK_DIR"
echo "dest : $DEST"
[ -f "$TRUNK_DIR/fedavg_state.pt" ] || { echo "no trunk checkpoint" >&2; exit 1; }
mkdir -p "$DEST"
[ -f "$DEST/fedavg_state.pt" ] || cp "$TRUNK_DIR/fedavg_state.pt" "$DEST/fedavg_state.pt"

python examples/fedpyhealth/train.py $COMMON --regime fedavg_ft --ft-epochs 20 \
    --adapter lora_head --resume
