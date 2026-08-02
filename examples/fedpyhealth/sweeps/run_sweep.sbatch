#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-sweep
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%A_%a.out

# SLURM ARRAY job for the FedAvg / FedAvg+FT hyperparameter sweep.
#
# Each array task runs exactly ONE config from the runlist: the line at index
# SLURM_ARRAY_TASK_ID (0-based). Tasks are independent single-A100 jobs, so the
# array runs them in parallel up to whatever concurrency the queue (and the %N
# throttle) allows. Every config has a distinct auto-derived run_name, so their
# checkpoints, TensorBoard logs, and _outputs/results/<run_name>.json never
# collide.
#
# 1) Generate the configs + runlist (prints the exact submit command):
#      .venv/bin/python examples/fedpyhealth/sweeps/make_sweep.py \
#          examples/fedpyhealth/sweeps/grid.yaml
#
# 2) Submit the array (0..N-1; %8 caps to 8 running at once -- raise/drop to taste):
#      mkdir -p _outputs/slurm
#      sbatch --array=0-23%8 examples/fedpyhealth/sweeps/run_sweep.sbatch \
#          _outputs/sweeps/fedavg_opt_v1/runlist.txt
#
# 3) After it finishes, rank the runs:
#      .venv/bin/python examples/fedpyhealth/sweeps/aggregate_results.py \
#          --sort Prevalence_R2 --desc
#
# Wall time: runtime ~ n_rounds * local_epochs. The 18h budget covers the default
# grid's heaviest cell (n_rounds=30, local_epochs=2, ~8h) with margin; raise
# --time if you widen local_epochs / n_rounds.

set -euo pipefail

RUNLIST="${1:?usage: sbatch run_sweep.sbatch <runlist.txt>}"
LINE=$((SLURM_ARRAY_TASK_ID + 1))
CONFIG="$(sed -n "${LINE}p" "$RUNLIST")"
if [ -z "${CONFIG}" ]; then
  echo "No config at line ${LINE} of ${RUNLIST} (array index out of range)" >&2
  exit 1
fi
echo "array task ${SLURM_ARRAY_TASK_ID} -> ${CONFIG}"

source .venv/bin/activate
python examples/fedpyhealth/ehr_eicu.py --config "${CONFIG}"
