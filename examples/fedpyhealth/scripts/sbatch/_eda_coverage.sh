#!/bin/bash
# Code-coverage EDA: of the codes a hospital needs but its OWN generator never
# emits, how many does federation supply? See notes/code-coverage-eda.md.
#
# Pure CPU -- no model is loaded, every arm's synthetic.json is already on disk.
# Takes a GPU only because Delta rejects zero-GPU jobs under the gpu account.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=eda-coverage
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
CACHE="$FEDCOHORT_CACHE/hilo8_random"

RUNS="--run local=_outputs/local_full_hilo8_random_nes_save
      --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_nes_adapterlora_head_save
      --run centralized=_outputs/centralized_full_hilo8_random_nes_save"

# The scored pool is what Test 2 reports on, so the two analyses answer about
# the same codes. `rare` repeats it on the strict global_rare subset, where the
# federation claim actually lives.
for poolset in scored rare; do
    python -u examples/fedpyhealth/scripts/eda_code_coverage.py \
        --cohort-cache "$CACHE" $RUNS \
        --fold test --code-pool "$poolset" --budget match --n-resamples 20 --min-carriers 1,5 \
        --out "_outputs/results/eda_coverage_${poolset}"
done
echo done
