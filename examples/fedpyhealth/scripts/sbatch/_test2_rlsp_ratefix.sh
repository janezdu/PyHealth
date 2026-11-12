#!/bin/bash
# RLSP for the two rate-penalty reruns (FedRep, LG-FedAvg + --rate-mu 1e-3) on
# every code set the utility panel shows, with the same selection flags as the
# panel's existing cells so the bars are directly comparable:
#
#   rand  --code-pool all  random draw, >= 20 test positives, seeds 0-2
#   top   --code-pool all  --eval-code-order top (fixed set, one run)
#   rare  --code-pool rare random draw, >= 20 test positives, seeds 0-2
#
# Only these two arms are trained here; the panel's reference lines come from
# the original files for each cell. Two per-hospital arms is ~42 classifiers
# per run (~8 min at ~10.6 s each), so the longest task (3 seeds) is ~25 min.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=t2-rlsp-ratefix
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run fedrep_rate=_outputs/fedrep_E2_R50_hilo8_random_privacy_rate0.001_nes_save
      --run lgfedavg_rate=_outputs/lg_fedavg_E2_R50_hilo8_random_privacy_rate0.001_nes_save"
for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

POOLS=(rand rand top top rare rare); MIXES=(natural upsample natural upsample natural upsample)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}; MIX=${MIXES[$SLURM_ARRAY_TASK_ID]}
case "$POOL" in
  rand) SEL="--code-pool all --eval-code-order random";  SEEDS="0 1 2" ;;
  top)  SEL="--code-pool all --eval-code-order top";     SEEDS="0" ;;
  rare) SEL="--code-pool rare --eval-code-order random"; SEEDS="0 1 2" ;;
esac
W=""; [ "$MIX" = upsample ] && W="--rlsp-weight 0.5 --rlsp-weight-mode upsample"

for seed in $SEEDS; do
    out="$RES/test2_rlspfix_${POOL}_${MIX}_rnn_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ pool=$POOL mix=$MIX seed=$seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp $W --classifier rnn --fold test --min-positives 20 \
        $SEL --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
done
echo done
