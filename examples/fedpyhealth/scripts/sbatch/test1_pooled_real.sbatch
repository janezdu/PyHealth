#!/bin/bash
# Test 1 with BOTH scopes pooled: every hospital's synthetic set scored against
# the same cohort-wide test fold, on the union rare pool.
#
#   --real-scope pooled : the target is the 2,430-patient pooled test fold, not
#       the site's own 40-523. Prevalence resolves only to 1/n_real, so at
#       hospital 429 the per-site version cannot express anything below 0.025 --
#       a hard floor on measurable fidelity that no amount of generated data
#       lifts. Pooling takes that to 1/2430.
#   --rare-scope pooled : the union of every site's rare set (548 codes),
#       intersected with the pooled fold -> 478 codes, the SAME set for every
#       hospital and identical to test2's scored pool.
#
# The question changes with the target: this asks how far each site's generator
# sits from the COHORT distribution, not from its own. For local and fedavg_ft
# that is a real per-site measurement. For fedavg and centralized -- one shared
# generator, one synthetic set copied to all eight sites -- all eight rows are
# the same comparison and will come out identical; that is correct, not a bug.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test1-pooled-real
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
    --real-scope pooled --rare-scope pooled \
    --out _outputs/results/tests/test1_prevalence_pooledreal_pooledrare.json
