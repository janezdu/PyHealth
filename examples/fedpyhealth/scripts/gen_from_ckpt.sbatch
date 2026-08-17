#!/bin/bash
# Re-sample a finished run's synthetic patients from its checkpoints.
#
# Sampling is minutes; training was hours. This job exists so changing how much
# synthetic data a run produces never drags an ~18h GPU job behind it.
#
#   sbatch --job-name=gen-centralized \
#          examples/fedpyhealth/scripts/gen_from_ckpt.sbatch \
#          _outputs/centralized_full_strat8_random_save
#
# The existing synthetic.json is backed up first: generate.py is newer than the
# data it overwrites, and the finished runs are not cheap to reproduce.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-generate
#SBATCH --time=00:45:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail

SAVE_DIR="${1:?usage: sbatch gen_from_ckpt.sbatch <save_dir> [extra generate.py args]}"
shift || true

source .venv/bin/activate

if [ -f "$SAVE_DIR/synthetic.json" ]; then
    cp -n "$SAVE_DIR/synthetic.json" "$SAVE_DIR/synthetic.json.bak"
    echo "backed up existing synthetic.json -> $SAVE_DIR/synthetic.json.bak"
fi

python examples/fedpyhealth/generate.py --save-dir "$SAVE_DIR" "$@"
