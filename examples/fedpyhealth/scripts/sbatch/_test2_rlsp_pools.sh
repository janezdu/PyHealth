#!/bin/bash
# RLSP for the utility panel's seven arms on two NEW code pools. The existing
# panel scores 30 codes drawn at random from the 91 with >= 20 test positives
# (code_pool=all), i.e. mostly common codes. These add:
#
#   rare  30 codes drawn from the 87 pooled-rare codes with >= 20 test
#         positives -- same threshold, so AUROC is as reliable as the existing
#         panel's. Three draws (seeds 0-2), as the existing panel uses.
#   top   the 30 codes with the MOST test positives (--eval-code-order top).
#         Fixed, not drawn, so one run per mix; --eval-seed would not change it.
#
# One array task per (pool, mix). Completed files are skipped on rerun, so a
# timeout loses only the seed in flight. Walltimes sized from t2-rlsp-all
# (~10.6 s/classifier, ~110 classifiers per seed for this arm set) with 2x
# margin -- the last time these were cut to 1h, five of six tasks timed out.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-pools
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run lat_lora_ft=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save
      --run ditto0.1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save
      --run perada=_outputs/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save
      --run fedprox=_outputs/fedprox_E2_R50_hilo8_random_privacy_prox0.01_nes_save
      --run fedrep=_outputs/fedrep_E2_R50_hilo8_random_privacy_nes_save
      --run lgfedavg=_outputs/lg_fedavg_E2_R50_hilo8_random_privacy_nes_save"
for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

POOLS=(rare rare top top); MIXES=(natural upsample natural upsample)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}; MIX=${MIXES[$SLURM_ARRAY_TASK_ID]}
if [ "$POOL" = rare ]; then
    SEL="--code-pool rare --eval-code-order random"; SEEDS="0 1 2"
else
    SEL="--code-pool all --eval-code-order top";     SEEDS="0"
fi
W=""; [ "$MIX" = upsample ] && W="--rlsp-weight 0.5 --rlsp-weight-mode upsample"

for seed in $SEEDS; do
    out="$RES/test2_rlsp${POOL}_${MIX}_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ pool=$POOL mix=$MIX seed=$seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp $W --classifier rnn --fold test --min-positives 20 \
        $SEL --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
