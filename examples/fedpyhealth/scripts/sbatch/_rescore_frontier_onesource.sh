#!/bin/bash
# Rescore EVERY arm on the frontier figure in ONE pass, from one source.
#
# The figure currently reads three scored files -- mix_*_prebuild (point
# estimates from compare_prevalence_metrics.py), test1_latent_*_b200 and
# test1_adapters20_*_b200 (200 bootstraps from test1_prevalence.py). Mixing
# scorers has already produced a visible disagreement once: the same checkpoint
# read +0.403 in one file and +0.570 in another. Every number on the figure
# should come from one invocation of one scorer.
#
# CPU ONLY, NO GENERATION. test1_prevalence does not touch CUDA, and every arm
# below was previously scored at --synth-cap 8000, so its synthetic.json
# already holds at least that many per hospital. Nothing needs regenerating,
# which is what makes this runnable on the (near-empty) cpu queue instead of
# waiting behind ~2000 GPU jobs.
#
# It also writes the manifest that has been missing: mix_*_prebuild records no
# arm->checkpoint mapping at all, so the provenance of the figure's oldest
# points is currently inference from directory names rather than a record.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=rescore-frontier
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run local=_outputs/local_full_hilo8_random_nes_save
      --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run centralized_rw=_outputs/centralized_full_hilo8_random_rw_save
      --run full_ft20=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapternone_save
      --run lora_head20=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_save
      --run fedavg_irm1e4=_outputs/fedavg_full_hilo8_random_irmrho10000_save
      --run fed_xm4_do03=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save
      --run do01_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.1_xm4_save
      --run lat_fed_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_z16_xm4_save
      --run lat_lora_ft=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save"

# The record that mix_*_prebuild never kept.
echo "$RUNS" | tr -s ' ' '\n' | grep -v '^--run$' | grep '=' \
    > _outputs/results/frontier_manifest.txt
echo "wrote _outputs/results/frontier_manifest.txt"

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="$RES/test1_frontier1src_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ scoring $tag (real-scope $scope)  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done
echo done
