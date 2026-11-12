#!/bin/bash
# RLSP for the six FL baselines that the 14-arm run left out.
#
# test2_rlspall_* covered ditto0.1 and perada only. These six have prevalence
# R^2 at 200 bootstraps but no downstream number at all, so the baseline table
# is half prevalence-only -- and three of them (fedprox, fedrep, lgfedavg) are
# the arms that LOOK broken on prevalence. Whether they are also broken
# downstream is the interesting question: their Pearson is normal (0.83-0.88),
# so the code ORDERING survives and only the scale is wrong. A downstream
# classifier may not care about scale at all.
#
# Both mixes, matching _test2_rlsp_all.sh exactly so these merge onto the same
# axis as the 14 arms already there. Regeneration first: all six hold 5000
# synthetic per hospital and --train-budget 8000 needs 8000.
#
# One task per arm: 6 x ~12 min serial is 70 min, but as an array each is ~12
# and a short task backfills far better on this partition.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-bl
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

PAIRS=(
 "ditto0=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_nes_save"
 "ditto1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto1_nes_save"
 "ditto10=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto10_nes_save"
 "fedprox=_outputs/fedprox_E2_R50_hilo8_random_privacy_prox0.01_nes_save"
 "fedrep=_outputs/fedrep_E2_R50_hilo8_random_privacy_nes_save"
 "lgfedavg=_outputs/lg_fedavg_E2_R50_hilo8_random_privacy_nes_save"
)
pair="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
name="${pair%%=*}"; dir="${pair#*=}"
[ -d "$dir" ] || { echo "missing $dir" >&2; exit 1; }

echo "############ $name: regenerating at 8000/hospital  ($(date +%H:%M:%S))"
python examples/fedpyhealth/generate.py --save-dir "$dir" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000

for mix in natural upsample; do
  W=""; [ "$mix" = "upsample" ] && W="--rlsp-weight 0.5 --rlsp-weight-mode upsample"
  for seed in 0 1 2; do
    out="_outputs/results/tests/test2_rlspbl_${name}_${mix}_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ $name $mix seed $seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" \
        --run "$name=$dir" --rlsp $W --classifier rnn \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
  done
done
echo done
