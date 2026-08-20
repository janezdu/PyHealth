#!/bin/bash
# How much of a training step is spent formatting input rather than learning?
#
#   sbatch examples/fedpyhealth/scripts/bench_encode.sbatch
#
# Before the multi-hot rewrite this reported, at batch 128 on an A100:
#   encode 13.887 s/batch (84.5%)   fwd+bwd 2.553 s/batch (15.5%)
# and in steady state, once CUDA warmup is excluded from batch 0, the
# transformer's own forward+backward was ~0.03 s -- encoding was ~99.8% of a
# step. Run it again after any change to _encode_visits or the processor.
#
# Needs a GPU to mean anything: the old encoder's cost was per-patient .item()
# syncs and single-element kernel launches, neither of which costs on CPU.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=enc-bench
#SBATCH --time=00:20:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
# Both batch sizes: 128 is what the in-flight runs use, 256 is the setting
# chosen for future runs. 12 batches so steady state has something to average
# once batch 0's CUDA warmup is excluded.
for bs in 128 256; do
  echo "############ batch_size=$bs"
  python examples/fedpyhealth/utils/capture_encode_golden.py \
      --cohort-cache $FEDCOHORT_CACHE/hilo8_random \
      --hospital 458 --batches 12 --batch-size $bs --device cuda --time-only
done
