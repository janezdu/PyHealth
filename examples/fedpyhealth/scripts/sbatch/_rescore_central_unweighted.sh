#!/bin/bash
# Score the UNWEIGHTED centralized arm with the Table 1 scorer settings.
#
# Table 1 reads test1_frontier1src_*, which only holds centralized_rw (the one
# arm in the table trained with rare-code upweighting). Every other row --
# local, FedAvg, the perFL baselines, FedLIBRA -- is unweighted, so this is the
# matched centralized reference. Same flags as _rescore_frontier_onesource.sh;
# the older b200 page scored this arm under a different setup, so its numbers
# are not comparable to the table.
#
# CPU only (pandas + sklearn). Submit from the repo root:
#   sbatch examples/fedpyhealth/scripts/sbatch/_rescore_central_unweighted.sh
# or run it with bash on any interactive node.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=rescore-centunw
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests
RUN="centralized=_outputs/centralized_full_hilo8_random_nes_save"

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="$RES/test1_centunw1src_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" \
        --run "$RUN" --fold test --synth-cap 8000 --rare-scope pooled \
        --real-scope "$scope" --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done

# Table 1 cells: mean over the 8 hospitals of the bootstrap point estimate.
python - <<'EOF'
import json
for tag in ("spec", "gen"):
    hs = json.load(open(f"_outputs/results/tests/test1_centunw1src_{tag}_b200.json"))["runs"]["centralized"]
    row = {m: sum(v[f"PrevVal_All_Prevalence_{m}"][0] for v in hs.values()) / len(hs)
           for m in ("Pearson", "R2")}
    print(f"{tag}: Pearson {row['Pearson']:.3f}  R2 {row['R2']:.3f}  ({len(hs)} hospitals)")
EOF
