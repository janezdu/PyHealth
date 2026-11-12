#!/bin/bash
# Score the latent-WIDTH sweep: z in {8,16,32,64} x K in {4,8}.
#
# Six runs (latent_width_v1-*) were trained on 2026-09-08 and NEVER SCORED, so
# "is z=16 the right width?" has no evidence behind it -- z=16 is simply the
# value every plotted latent arm happens to use. This closes that.
#
# The grid is IRM-OFF and dropout 0 throughout, which is why lat_xm4 (z=16,
# K=4, IRM off, trained 2026-09-18) is the matching centre point and NOT
# lat_fed_irm -- that one carries IRM rho=100 and would make width and IRM
# change together. lat_xm4 is scored by the xm-fill job; it is included here
# too so one file holds the whole curve, and the two independent estimates of
# the same arm are a free consistency check.
#
# HOLE IN THE GRID: z=16 K=8 was never trained. So K=8 has z in {8,32,64} only,
# and the K=4 row is the one to read for width.
#
# ONE ARM PER ARRAY TASK. The serial version was 7 arms x ~8 min back to back;
# scoring an arm is independent of the others (each is compared against the real
# fold, with a fixed --bootstrap-seed 0), so they parallelise exactly. Wall time
# drops from ~70 min to ~10, and a 1h task backfills on a 2000-deep partition
# where a 3h one waits. Each task writes its own file; plot_rw_frontier takes
# repeated --spec/--gen, so no merge step is needed.
#
# Synthetic is regenerated at 8000/hospital before scoring, matching
# _score_latent.sh: --synth-cap 8000 fails outright if a run generated fewer,
# and these predate that convention.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=score-zwidth
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

declare -a PAIRS=(
  "z8k4=_outputs/latent_width_v1-000-8-4_save"
  "z8k8=_outputs/latent_width_v1-001-8-8_save"
  "z16k4=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_z16_xm4_save"
  "z32k4=_outputs/latent_width_v1-002-32-4_save"
  "z32k8=_outputs/latent_width_v1-003-32-8_save"
  "z64k4=_outputs/latent_width_v1-004-64-4_save"
  "z64k8=_outputs/latent_width_v1-005-64-8_save"
)

pair="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
name="${pair%%=*}"; dir="${pair#*=}"
[ -d "$dir" ] || { echo "missing $dir" >&2; exit 1; }

echo "############ $name: regenerating at 8000/hospital  ($(date +%H:%M:%S))"
python examples/fedpyhealth/generate.py --save-dir "$dir" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000

for scoped in "hospital:spec" "pooled:gen"; do
    scope="${scoped%%:*}"; tag="${scoped##*:}"
    out="_outputs/results/tests/test1_zwidth_${name}_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ $name: scoring $tag (real-scope $scope)  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" \
        --run "$name=$dir" \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done
echo done
