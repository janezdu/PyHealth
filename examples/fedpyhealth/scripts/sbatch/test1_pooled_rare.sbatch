#!/bin/bash
# Test 1 with PrevVal_Rare_* scored on one shared definition of "rare".
#
# Default scope is per-hospital: each site's own <=5% codes, which is 296 codes
# at 458 and 103 at 429 -- a different tail at every site, so the per-hospital
# rare numbers are not strictly comparable to one another.
#
# --rare-scope pooled uses the union instead (548 codes, rare at >=1 hospital),
# intersected with the codes each site actually has in the test fold. The
# intersection is required, not a convenience: 79% of the pooled pool has zero
# prevalence at 429 and 83% at 358, and scoring those would add hundreds of
# (real 0, synthetic ~0) points that agree trivially, inflating Pearson and R^2
# while measuring nothing about the tail.
#
# --synth-cap 8000 as before, so all four arms are read at one resolution.
# CPU-only: test1 reads synthetic.json and computes prevalences.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test1-pooled-rare
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
    --fold test --synth-cap 8000 --rare-scope pooled \
    --out _outputs/results/tests/test1_prevalence_n8000_capped_pooledrare.json
