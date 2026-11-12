#!/bin/bash
# Score FedAvg + IRM (rho=1e2, 17-round warmup) with the Table 1 scorer settings,
# for the ablation table. This is FedLIBRA's IRM setting without best-of-K or
# fine-tuning; its rho=1e4 twin is already in test1_frontier1src_*. Same flags as
# _rescore_frontier_onesource.sh so the numbers sit next to the other rows.
#
# CPU only (pandas + sklearn). Submit from the repo root:
#   sbatch examples/fedpyhealth/scripts/sbatch/_rescore_irm1e2.sh
# or run it with bash on any interactive node.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=rescore-irm100
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests
RUN="fedavg_irm1e2=_outputs/fedavg_full_hilo8_random_irmrho100_save"

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="$RES/test1_irm1e2_1src_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" \
        --run "$RUN" --fold test --synth-cap 8000 --rare-scope pooled \
        --real-scope "$scope" --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done

# Table 1 cells: mean over the 8 hospitals of the bootstrap point estimate.
python - <<'EOF'
import json
for tag in ("spec", "gen"):
    hs = json.load(open(f"_outputs/results/tests/test1_irm1e2_1src_{tag}_b200.json"))["runs"]["fedavg_irm1e2"]
    row = {m: sum(v[f"PrevVal_All_Prevalence_{m}"][0] for v in hs.values()) / len(hs)
           for m in ("Pearson", "R2")}
    print(f"{tag}: Pearson {row['Pearson']:.3f}  R2 {row['R2']:.3f}  ({len(hs)} hospitals)")
EOF
