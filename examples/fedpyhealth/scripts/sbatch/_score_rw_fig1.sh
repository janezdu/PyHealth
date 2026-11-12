#!/bin/bash
# All six Figure 1 arms under the SAME treatment: rare-code upweighting on,
# early stopping on.
#
# Zda4's Q2: "Why is upweighting used only for the pooled reference? Please
# retrain all methods with it." The 2x2 shows the asymmetry is worse than that
# question assumes -- the published centralized row is the upweighted,
# early-stopped cell, while the published FedAvg and local-only rows are the
# unweighted, full-budget cell. The reference got BOTH treatments and the
# others got neither.
#
# Nothing is trained or generated here: all six upweighted runs already exist
# (three from Nov 10, three older), all at profile full, R=50, E=2, early_stop
# true, rare_upweight true, all with >= 8000 synthetic per hospital. Checked
# before writing this.
#
# CPU ONLY.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=score-rw-fig1
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

RUNS="--run rw_local=_outputs/local_full_hilo8_random_rw_save
      --run rw_centralized=_outputs/centralized_full_hilo8_random_rw_save
      --run rw_fedavg=_outputs/fedavg_full_hilo8_random_rw_save
      --run rw_ft_full=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_rw_save
      --run rw_ft_lora=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_rw_lora_headr8_save
      --run rw_fedlibra=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_rw_irm100w17_z16_xm4_lora_headr8_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

echo "############ Test 1, SPECIALIST, 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_rwfig1_spec_b200.json

echo "############ Test 1, GENERALIST, 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_rwfig1_gen_b200.json
echo done
