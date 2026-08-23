#!/bin/bash
# Test 1 with every arm read at the SAME prevalence resolution.
#
# Prevalence resolves only to 1/N, so an arm holding more synthetic patients
# scores a better R2 for reasons unrelated to its generator. The shared-generator
# regimes (fedavg, centralized) have no per-site ceiling -- train.py hands each
# hospital the whole pooled generation -- so at --synth-per-hospital 8000 they
# carry 64,000 per hospital while local and fedavg_ft carry 8,000. Capping at
# 8,000 puts all four on one grid, which is the only way the cross-arm numbers
# mean anything.
#
# CPU only: test1 reads synthetic.json and computes prevalences. No model, no GPU.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=test1-capped8000
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --run fedavg=_outputs/fedavg_full_hilo8_random_save \
    --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_save \
    --run centralized=_outputs/centralized_full_hilo8_random_save \
    --run local=_outputs/local_full_hilo8_random_save \
    --fold test --synth-cap 8000 \
    --out _outputs/results/tests/test1_prevalence_n8000_capped.json
