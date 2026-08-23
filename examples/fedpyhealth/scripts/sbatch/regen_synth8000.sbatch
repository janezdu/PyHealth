#!/bin/bash
# Regenerate every arm's synthetic patients at 8,000 per hospital, then re-score
# Test 1.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/regen_synth8000.sbatch
#
# WHY 8,000. Test 1 compares each hospital's synthetic code prevalences against
# that hospital's real test fold, and a set of N patients can only express
# prevalences in multiples of 1/N. At the current 2,000 the rarest scored code
# at the four largest sites is expected to appear ~1.5-1.8 times, so its
# "prevalence" is 1/2000 or 2/2000 and the rare-code Pearson there is reading
# rounding rather than fidelity. 8,000 puts every site at >=6 expected
# occurrences. (The documented 10-occurrence bar needs ~13,000 at hospital 458;
# 8,000 is the compromise, so band the deep tail by support before claiming it.)
#
# No retraining: generate.py samples from the checkpoints already on disk.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=regen-synth8000
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

PER_HOSP=8000
ARMS="fedavg fedavg_ft centralized local"

# Keep the 2,000-per-hospital sets. generate.py overwrites synthetic.json in
# place, and the current test1/test2 results were computed from those files --
# without this the existing numbers become unreproducible. Guarded so a re-run
# of this script does not overwrite the backup with the 8,000 version.
for a in $ARMS; do
  d="_outputs/${a}_full_hilo8_random_save"
  if [ -f "$d/synthetic.json" ] && [ ! -f "$d/synthetic_perhosp2000.json" ]; then
    cp "$d/synthetic.json" "$d/synthetic_perhosp2000.json"
    echo "backed up $d/synthetic.json -> synthetic_perhosp2000.json"
  fi
done

for a in $ARMS; do
  echo "############ regenerating $a at ${PER_HOSP}/hospital"
  python examples/fedpyhealth/generate.py \
      --save-dir "_outputs/${a}_full_hilo8_random_save" \
      --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
      --synth-per-hospital $PER_HOSP
done

# Separate --out so the 2,000 and 8,000 scores sit side by side rather than one
# silently replacing the other.
echo "############ Test 1 at ${PER_HOSP}/hospital"
python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --run fedavg=_outputs/fedavg_full_hilo8_random_save \
    --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_save \
    --run centralized=_outputs/centralized_full_hilo8_random_save \
    --run local=_outputs/local_full_hilo8_random_save \
    --fold test \
    --out _outputs/results/tests/test1_prevalence_n8000.json
