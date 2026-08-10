#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-ehr-test2
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out

# Test 2: pooled rare-code ML efficacy (TSTR) across every regime.
# Drives examples/fedpyhealth/rare_code_efficacy.py.
#
# Submit AFTER the four training runs finish -- it consumes the synthetic.json
# each run persists under _outputs/<run_name>_save/. One job scores all four
# regimes plus the reference arms (prior floor, real_local, real_pooled_budgeted
# and the real_pooled ceiling), so eICU is loaded once rather than four times.
#
# Submit from the PyHealth repo root:
#   sbatch examples/fedpyhealth/run_rare_efficacy.sh \
#       --cohort-file examples/fedpyhealth/cohorts/rare8_v2.json \
#       --mask-folds 10 \
#       --run centralized=_outputs/<centralized_run_name>_save \
#       --run local=_outputs/<local_run_name>_save \
#       --run fedavg=_outputs/<fedavg_run_name>_save \
#       --run fedavg_ft=_outputs/<fedavg_ft_run_name>_save
#
# Chain it off the training jobs instead of waiting:
#   sbatch --dependency=afterok:<id1>:<id2>:<id3>:<id4> \
#       examples/fedpyhealth/run_rare_efficacy.sh --run ...
#
# COST. Every arm trains one classifier per hospital per mask fold, so the job
# size is (10 reference + 8 x n_regimes) x --mask-folds trainings. Four regimes
# at --mask-folds 10 is 420 classifiers -- which sounds worse than it is: the
# 10 reference arms x 10 folds measured 3m09s on a laptop CPU, so the full grid
# is well under an hour on a GPU node and the 06:00:00 above is generous. Use
# --mask-folds 1 only for a smoke test: it strips the whole tail at once and
# leaves no rare-to-rare co-occurrence for any arm to learn from.
#
# Results land in _outputs/results/test2_rare_efficacy.json (gitignored).

set -euo pipefail

source .venv/bin/activate

python examples/fedpyhealth/rare_code_efficacy.py "$@"
