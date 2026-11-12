#!/bin/bash
# Score the three arms sweep_rw_fill.sh added to the upweighted (rwfig) figure,
# with the same flags as test1_rwfig_*_b200, then redraw the six-arm figure.
#
# CPU only. From the repo root:
#   sbatch examples/fedpyhealth/scripts/sbatch/_rescore_rwfill.sh
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=rescore-rwfill
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests
O=_outputs
RUNS="--run full_ft20=$O/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_rw_save
      --run lora_head20=$O/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_rw_lora_headr8_save
      --run lat_lora_ft=$O/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_rw_irm100w17_z16_xm4_lora_headr8_save"

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="$RES/test1_rwfill_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done

# Six-arm figure: local/centralized/fedavg from rwfig + the three new arms.
python examples/fedpyhealth/scripts/plot_rw_frontier.py \
    --spec $RES/test1_rwfig_spec_b200.json --spec $RES/test1_rwfill_spec_b200.json \
    --gen  $RES/test1_rwfig_gen_b200.json  --gen  $RES/test1_rwfill_gen_b200.json \
    --only local,centralized,fedavg,full_ft20,lora_head20,lat_lora_ft \
    --rename "lat_lora_ft=FedLIBRA" \
    --title "All arms trained with rare-code upweighting" \
    --out _outputs/results/rw_frontier_6arm.png
