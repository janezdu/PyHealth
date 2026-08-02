#!/bin/bash
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=eicu-freeze-split
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out

# Freeze the 8-hospital cohort with rare-code-stratified 80/20 splits.
# Drives examples/fedpyhealth/freeze_cohort_split.py. Run this ONCE; every
# regime (centralized / local / fedavg / fedavg_ft) then loads the manifest so
# all four compare on byte-identical data.
#
# Submit from the PyHealth repo root:
#   mkdir -p _outputs/slurm   # --output dir must exist before submit
#   sbatch examples/fedpyhealth/run_freeze_split.sh
#
# Extra flags pass straight through, e.g. a smoke manifest:
#   sbatch examples/fedpyhealth/run_freeze_split.sh --dev \
#       --rare-prevalence-max 0.5 \
#       --out examples/fedpyhealth/cohorts/rare8_smoke.json
#
# NOTE on --dev: it caps eICU to ~1000 patients, so a hospital holds ~20 of
# them. At the default --rare-prevalence-max 0.05 a code would need <= 1 patient
# to be rare while needing >= 2 to qualify -- an empty rare set. The script
# raises rather than silently producing an unstratified split, but you must pass
# a larger threshold (0.5) for any --dev run to work at all.
#
# This job needs no GPU -- it loads eICU and counts, all on CPU. It requests one
# anyway because Delta rejects ANY zero-GPU job under a *-delta-gpu account.
# gpuA40x4 is the cheapest tier. Budget ~1-2h cold (dominated by the eICU load
# and the single streaming pass over every sample).
#
# Outputs:
#   cohorts/rare8_v1.json          full manifest WITH patient ids -- gitignored
#   cohorts/rare8_v1.summary.json  aggregates + hashes only -- safe to commit

set -euo pipefail

source .venv/bin/activate

# Pre-flight: validate the stratifier before paying for the eICU load. Pure
# numpy, runs in under a second, and turns "the 2h job died at the end" into
# "the job died at the start".
#
# This deliberately does NOT use pytest -- the Delta venv has no pytest, and a
# missing module here would kill the job under `set -e` before it does any work.
# tests/core/test_stratified_split.py covers the same invariants for `make test`.
echo "=== pre-flight: stratifier self-test ==="
python examples/fedpyhealth/freeze_cohort_split.py --self-test

echo "=== freezing cohort ==="
python examples/fedpyhealth/freeze_cohort_split.py "$@"
