#!/bin/bash
# Score the four federated Figure 1 arms at n=3 independent training runs each,
# so the figure can carry error bars.
#
# rep1 of each arm is the PUBLISHED run -- the same directory the current
# figure point comes from -- so the spread is around the number already in the
# paper rather than around a fresh set. reps 2 and 3 come from _reps_fed4.sh.
#
# Aggregate as the main table does: mean over the 8 hospitals of
# PrevVal_All_Prevalence_R2, then mean and spread over the three runs. The
# other three plausible statistics (median, rare-only) disagree by more than
# most arm-to-arm gaps, so do not mix them.
#
# CPU ONLY. Masked CUDA so this cannot take a device.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=score-reps-fed4
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

RUNS="--run fedavg_rep1=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_rep2=_outputs/rep2_fedavg_save
      --run fedavg_rep3=_outputs/rep3_fedavg_save
      --run ft_full_rep1=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapternone_save
      --run ft_full_rep2=_outputs/rep2_ft_full_save
      --run ft_full_rep3=_outputs/rep3_ft_full_save
      --run ft_lora_rep1=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_save
      --run ft_lora_rep2=_outputs/rep2_ft_lora_save
      --run ft_lora_rep3=_outputs/rep3_ft_lora_save
      --run fedlibra_rep1=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_irm100w17_nes_z16_xm4_lora_headr8_save
      --run fedlibra_rep2=_outputs/rep2_fedlibra_save
      --run fedlibra_rep3=_outputs/rep3_fedlibra_save"

for d in $(echo "$RUNS" | grep -o '_outputs/[^ ]*_save'); do
    [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
done

echo "############ Test 1, SPECIALIST, 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_repsfed4_spec_b200.json

echo "############ Test 1, GENERALIST, 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_repsfed4_gen_b200.json
echo done
