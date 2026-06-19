#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-ehr-eicu-full
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out

# Federated generative-EHR example: FULL eICU -> per-hospital FedAvg HALO -> metrics.
# Drives the self-contained example examples/fedpyhealth/ehr_eicu.py at production
# scale (full dataset, many hospital clients, many FedAvg rounds, larger model).
#
# This wrapper differs from run_ehr_eicu.sh only in its SBATCH budget (more mem,
# 24h wall time) and the --profile full flag. FedAvg here is single-GPU (clients
# trained sequentially), so one A100 is enough; raise --time/--mem if you scale
# --max-hospitals or --n-rounds well past the defaults.
#
# Submit from the PyHealth repo root:
#   mkdir -p _outputs/slurm   # --output dir must exist before submit
#   sbatch examples/fedpyhealth/run_ehr_eicu_full.sh
#
# Per-knob overrides pass straight through, e.g.:
#   sbatch examples/fedpyhealth/run_ehr_eicu_full.sh --n-rounds 100 --max-hospitals 100
#
# Checkpoint/resume: every round writes _outputs/halo_fed_full_save/fedavg_state.pt.
# If a job hits the wall, resubmit with --resume to continue from the last round:
#   sbatch examples/fedpyhealth/run_ehr_eicu_full.sh --resume --n-rounds 100 --max-hospitals 100
# (use the SAME --max-hospitals/--min-hospital-samples; resume refuses a mismatched
# partition.) Chain across windows with: sbatch --dependency=afterany:<jobid> ... --resume
#
# All run artifacts (slurm logs + HALO checkpoints) land under _outputs/, which
# is gitignored.

source .venv/bin/activate

python examples/fedpyhealth/ehr_eicu.py --profile full "$@"
