#!/bin/bash
# Put the rare-upweighted runs on exactly the same footing as the unweighted
# ones, so a difference between them is the upweighting and nothing else.
#
# As trained, the _rw runs differ from the unweighted comparison in THREE ways
# at once: they generated 2,000 patients per hospital (the profile default)
# against the unweighted runs' regenerated 8,000; their test1 used the
# per-hospital rare definition rather than the shared union; and it applied no
# --synth-cap. Any one of those could produce a difference on its own, so the
# arms are not comparable until all three are matched.
#
# This regenerates every _rw arm at 8,000/hospital from checkpoints already on
# disk (no retraining), then scores test1 with the identical flags used for the
# unweighted files: --synth-cap 8000 --rare-scope pooled, once against each
# site's own fold and once against the pooled cohort fold.
#
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=rw-regen-rescore
#SBATCH --time=01:30:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

ARMS="fedavg fedavg_ft centralized local"
for a in $ARMS; do
  d="_outputs/${a}_full_hilo8_random_rw_save"
  if [ -f "$d/synthetic.json" ] && [ ! -f "$d/synthetic_perhosp2000.json" ]; then
    cp "$d/synthetic.json" "$d/synthetic_perhosp2000.json"
  fi
  echo "############ regenerating ${a}_rw at 8000/hospital"
  python examples/fedpyhealth/generate.py \
      --save-dir "$d" --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
      --synth-per-hospital 8000
done

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_rw_save \
      --run fedavg_ft=_outputs/fedavg_ft_full_hilo8_random_rw_save \
      --run centralized=_outputs/centralized_full_hilo8_random_rw_save \
      --run local=_outputs/local_full_hilo8_random_rw_save"

echo "############ test1 RW, own fold"
python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled \
    --out _outputs/results/tests/test1_RW_n8000_capped_pooledrare.json

echo "############ test1 RW, pooled fold"
python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" $RUNS \
    --fold test --synth-cap 8000 --real-scope pooled --rare-scope pooled \
    --out _outputs/results/tests/test1_RW_pooledreal_pooledrare.json
