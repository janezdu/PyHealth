#!/bin/bash
# Rebuttal T2.1 (RS1V Q1): is the IRM gain just calibration?
#
# Regenerate FedAvg, centralized_rw and FedLIBRA at a grid of sampling
# temperatures. No retraining. generate.py writes synthetic_tau<T>.json beside
# the untouched tau=1 synthetic.json. Picking tau per hospital on val and
# scoring come afterwards (test1 currently reads synthetic.json only).
#
# From the repo root:
#   sbatch examples/fedpyhealth/scripts/sbatch/_rebut_tau_regen.sh
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=rebut-tau
#SBATCH --time=08:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
O=_outputs
ARMS="$O/fedavg_full_hilo8_random_nes_save
      $O/centralized_full_hilo8_random_rw_save
      $O/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save"

for dir in $ARMS; do
    for tau in 0.8 0.9 1.1 1.25; do
        [ -f "$dir/synthetic_tau${tau}.json" ] && { echo "skip $dir tau=$tau"; continue; }
        echo "############ $dir tau=$tau  ($(date +%H:%M:%S))"
        python examples/fedpyhealth/generate.py --save-dir "$dir" \
            --cohort-cache "$CACHE" --synth-per-hospital 8000 --temperature "$tau"
    done
done
echo done
