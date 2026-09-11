#!/bin/bash
# Score the three IDENTICAL-CONFIG runs. Their spread is TRAINING variance,
# which nothing on the plot has ever accounted for -- every noise figure so far
# comes from scoring one model twice, not from training the same thing twice.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=score-varrep
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
ORIG=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.1_xm4_save
RUNS=""
add () { [ -d "$2" ] || { echo "missing $2" >&2; return; }
  python examples/fedpyhealth/generate.py --save-dir "$2" \
      --cohort-cache "$CACHE" --synth-per-hospital 8000
  RUNS="$RUNS --run $1=$2"; }
add rep1 "$ORIG"
add rep2 _outputs/varrep2_fedavg_irm100w17_do0.1_xm4_save
add rep3 _outputs/varrep3_fedavg_irm100w17_do0.1_xm4_save
for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "_outputs/results/tests/test1_varrep_${tag}_b200.json"
done
