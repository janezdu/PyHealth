#!/bin/bash
# Rebuttal T0.3 + T0.4: score already-trained arms with the Table 1 scorer.
#
#   T0.3 centralized de-confound -- plain (ES), rw (upweight, ES), nes (no ES),
#        nrounds100. rw+nes was never trained, so the 2x2 has one empty cell.
#        nrounds80 has no synthetic.json and is skipped.
#   T0.4 IRM+LoRA without best-of-K (dob9 Q2), for Table 4.
#
# Same flags as _rescore_frontier_onesource.sh, so rows sit next to Table 1.
# CPU only. From the repo root:
#   sbatch examples/fedpyhealth/scripts/sbatch/_rebut_t0_rescore.sh
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=rebut-t0
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests
O=_outputs
RUNS="--run cent_plain=$O/centralized_full_hilo8_random_save
      --run cent_rw=$O/centralized_full_hilo8_random_rw_save
      --run cent_nes=$O/centralized_full_hilo8_random_nes_save
      --run cent_nr100=$O/centralized_full_hilo8_random_nrounds100_save
      --run irm_lora_nok=$O/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_irmrho100_irmwarmup34_save"

for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    out="$RES/test1_rebutt0_${tag}_b200.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ scoring $tag (real-scope $scope)  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 --out "$out"
done

# Table cells: mean over the 8 hospitals of the bootstrap point estimate.
python - <<'EOF'
import json
for tag in ("spec", "gen"):
    runs = json.load(open(f"_outputs/results/tests/test1_rebutt0_{tag}_b200.json"))["runs"]
    for name, hs in runs.items():
        row = {m: sum(v[f"PrevVal_All_Prevalence_{m}"][0] for v in hs.values()) / len(hs)
               for m in ("Pearson", "R2")}
        print(f"{tag:4s} {name:14s} Pearson {row['Pearson']:.3f}  R2 {row['R2']:.3f}")
EOF
