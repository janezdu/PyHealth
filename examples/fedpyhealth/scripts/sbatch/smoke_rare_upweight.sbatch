#!/bin/bash
# Five-minute smoke of --rare-upweight across all four regimes.
#
# The weighting touches shared library code (HALOModel.forward, train_model) and
# all four training helpers. A typo in any of those paths surfaces only when
# that regime runs, and the fedavg_ft path in particular is not reached until
# after the federated rounds finish -- which on the full profile is 20 minutes
# in, after a queue wait. The tiny profile exercises every path in about a
# minute of compute each.
#
# Tiny gives no usable numbers by design (num_synth=64). It answers "does this
# run", never "is this any good".
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-rw
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
for R in fedavg fedavg_ft centralized local; do
  echo "############ smoke $R (tiny, rare-upweight)"
  python examples/fedpyhealth/train.py --profile tiny --regime "$R" \
      --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
      --rare-upweight --no-tb
done
echo "############ ALL FOUR REGIMES OK"
