#!/bin/bash
# Fold 29 more arms into the SINGLE-SOURCE scored file.
#
# test1_frontier1src_* holds the 10 arms on the frontier figure. This adds the
# rest of what is already trained and has synthetic on disk, so every figure in
# the project can read one scorer at one setting instead of four files:
#
#   FL baselines   ditto x4, perada, fedprox, fedrep, lg_fedavg   (the 12-arm figure)
#   ouroboros      st0/10/20/30 + IRM / best-of-K add-ons
#   adapter family lora_attn20, last_mlp20, and the IRM-trunk versions
#   best-of-K      K=2/4/8 and the dropout-only controls that undercut them
#   varrep2/3      SAME CONFIG as do01_fed_irm, different training draw
#
# varrep2/varrep3 are the important ones. Every noise figure quoted on this
# project comes from scoring one model repeatedly; TRAINING variance has never
# been measured. These two are re-runs of the do01_fed_irm recipe, so the
# spread across {do01_fed_irm, varrep2, varrep3} is the first real estimate of
# it -- and it sets the bar for which gaps on the frontier figure mean anything.
#
# CPU ONLY. Nothing is regenerated; every arm below already holds >= 8000
# synthetic per hospital from an earlier scoring pass. The 10-arm run took
# ~5 minutes, so expect ~15 for this.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=rescore-rest
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests

RUNS="--run ditto0=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_nes_save --run ditto0.1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save --run ditto1=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto1_nes_save --run ditto10=_outputs/ditto_E2_R50_hilo8_random_privacy_ft2_ditto10_nes_save --run perada=_outputs/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save --run fedprox=_outputs/fedprox_E2_R50_hilo8_random_privacy_prox0.01_nes_save --run fedrep=_outputs/fedrep_E2_R50_hilo8_random_privacy_nes_save --run lgfedavg=_outputs/lg_fedavg_E2_R50_hilo8_random_privacy_nes_save --run st0=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_st0.5@1_save --run st10=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_st0.5r10@1_save --run st20=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_st0.5r20@1_save --run st30=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_st0.5r30@1_save --run st20_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_st0.5r20@1_save --run st20_xm4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_st0.5r20@1_save --run st20_xm4_irm=_outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.3_xm4_st0.5r20@1_save --run lora_attn20=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_attn_save --run last_mlp20=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlast_mlp_save --run irm_lora_head=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_irmrho100_irmwarmup34_save --run irm_lora_attn=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_attn_irmrho100_irmwarmup34_save --run irm_full_ft=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapternone_irmrho100_irmwarmup34_save --run xm4_cent=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save --run xm2_cent=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm2_save --run xm8_cent=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save --run xm8_fedavg=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save --run do03_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_save --run do01_only=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.1_save --run do01_fed_k4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.1_xm4_save --run varrep2=_outputs/varrep2_fedavg_irm100w17_do0.1_xm4_save --run varrep3=_outputs/varrep3_fedavg_irm100w17_do0.1_xm4_save"

echo "$RUNS" | tr -s ' ' '\n' | grep '=' >> _outputs/results/frontier_manifest.txt

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="$RES/test1_rest1src_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ scoring $tag (real-scope $scope)  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done
echo done
