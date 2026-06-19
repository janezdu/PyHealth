#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-ehr-eicu
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out

# Federated generative-EHR example: eICU subset -> per-hospital FedAvg HALO -> metrics.
# Drives the self-contained example examples/fedpyhealth/ehr_eicu.py.
#
# Submit from the PyHealth repo root:
#   mkdir -p _outputs/slurm   # --output dir must exist before submit
#   sbatch examples/fedpyhealth/run_ehr_eicu.sh
#
# All run artifacts (slurm logs + HALO checkpoints) land under _outputs/, which
# is gitignored (matched by the "*output*" rule in the repo .gitignore).
#
# This is the SMOKE launcher: a quick dev run (tiny dataset/model, ~minutes on
# one GPU). The run scale is selected by ehr_eicu.py's --profile flag; per-knob
# overrides (e.g. --n-rounds, --max-hospitals) can be appended after it. For a
# production-scale run with a larger SBATCH budget, use run_ehr_eicu_full.sh.

source .venv/bin/activate

python examples/fedpyhealth/ehr_eicu.py --profile smoke "$@"
