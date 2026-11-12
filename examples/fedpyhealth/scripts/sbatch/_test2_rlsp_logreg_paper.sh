#!/bin/bash
# Rebuttal C6: rerun the paper's utility table (Table: downstream) with logreg.
#
# The RNN real_local row (0.420 AUROC) is below chance at EVERY site, including
# the 1,500-1,800-record ones, while logreg on identical data, codes and folds
# scores 0.644 (test2_head_rlsp_seed*). So 0.420 is an RNN artifact (unweighted
# BCE, 10 epochs, no validation), not small hospitals, and every RNN "gain over
# real_local" is measured against it. logreg (class_weight=balanced) is the
# sound control; this puts every paper arm on it.
#
# Same draws/flags as _test2_rlsp_all.sh and _test2_rlsp_baselines.sh; only the
# classifier changes. CPU only. From the repo root:
#   sbatch examples/fedpyhealth/scripts/sbatch/_test2_rlsp_logreg_paper.sh
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=t2-logreg-paper
#SBATCH --time=08:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests
O=_outputs

RUNS="--run fedlibra=$O/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save
      --run fedavg=$O/fedavg_full_hilo8_random_nes_save
      --run ditto0.1=$O/ditto_E2_R50_hilo8_random_privacy_ft2_ditto0.1_nes_save
      --run perada=$O/perada_E2_R50_hilo8_random_privacy_ft2_kd1_nes_lora_headr8_save
      --run fedprox=$O/fedprox_E2_R50_hilo8_random_privacy_prox0.01_nes_save
      --run fedrep=$O/fedrep_E2_R50_hilo8_random_privacy_nes_save
      --run lgfedavg=$O/lg_fedavg_E2_R50_hilo8_random_privacy_nes_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

for mix in upsample natural; do
  W=""; [ "$mix" = "upsample" ] && W="--rlsp-weight 0.5 --rlsp-weight-mode upsample"
  for seed in 0 1 2; do
    out="$RES/test2_rlsppaper_${mix}_logreg_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "############ logreg $mix seed $seed  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
        --rlsp $W --classifier logreg \
        --fold test --code-pool all --min-positives 20 \
        --n-eval-codes 30 --eval-seed "$seed" \
        --train-budget 8000 --out "$out"
  done
done

# Macro AUROC per arm family, mean over hospitals and seeds.
python - <<'EOF'
import json, glob, statistics as st
for mix in ("upsample", "natural"):
    fam = {}
    for f in sorted(glob.glob(f"_outputs/results/tests/test2_rlsppaper_{mix}_logreg_seed*.json")):
        for k, v in json.load(open(f))["arms"].items():
            key = ":".join(k.split(":")[:2]) if k.startswith("rlsp") else k.split(":")[0]
            x = v["overall"]["roc_auc_macro"]
            if x == x:
                fam.setdefault(key, []).append(x)
    print(f"\n{mix}")
    for k, v in sorted(fam.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"  {k:22s} AUROC {st.mean(v):.3f}  (n={len(v)})")
EOF
