#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=eicu-hospital-stats
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out

# Per-hospital eICU EDA: hospital count, records-per-hospital distribution, the
# 8 smallest hospitals, their ICD-9 code histograms, and their rare codes.
# Drives examples/fedpyhealth/hospital_stats.py.
#
# Submit from the PyHealth repo root:
#   mkdir -p _outputs/slurm   # --output dir must exist before submit
#   sbatch examples/fedpyhealth/run_hospital_stats.sh
#
# Extra flags pass straight through, e.g.:
#   sbatch examples/fedpyhealth/run_hospital_stats.sh --n-smallest 12
#
# This job needs no GPU -- it only loads eICU and counts, all on CPU. It requests
# one anyway because Delta rejects ANY zero-GPU job submitted under a *-delta-gpu
# account ("Jobs requesting no GPU resources must be submitted under CPU type
# accounts"), on the `cpu` partition and on GPU partitions alike, and bgyw/bejr
# only hold gpu accounts. Verified with `sbatch --test-only`. The GPU sits idle;
# gpuA40x4 is used as the cheapest tier (and currently the shortest queue).
# If a CPU allocation ever lands, this becomes a genuine CPU job:
#   #SBATCH --partition=cpu --account=<your>-delta-cpu   (and drop --gpus-per-node)
#
# Runtime is dominated by the one-time streaming pass over every sample; budget
# ~1h cold. Afterwards the JSON cache makes re-reports instant on the login node:
#   python examples/fedpyhealth/hospital_stats.py --from-cache --rare-max-hospitals 2
#
# Outputs land under _outputs/eda/ (gitignored). They are hospital-level
# aggregates only -- no patient rows -- so they are safe to paste into notes.

source .venv/bin/activate

python examples/fedpyhealth/hospital_stats.py "$@"
