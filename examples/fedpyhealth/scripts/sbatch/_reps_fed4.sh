#!/bin/bash
# Independent training runs for the four federated arms on Figure 1, so the
# figure can carry error bars instead of single points.
#
# WHY THIS IS NOT A SEED SWEEP. train.py hardcodes SEED = 0 (line 74, "covers
# weight init only") and has no --seed flag. It does not need one: varrep2 and
# varrep3 differ from their base run in `run_name` and NOTHING else -- same
# seed, same config -- and scored R2 -0.025 / -0.008 / +0.007. Run-to-run
# variation is already there, from GPU nondeterminism rather than from the
# seed. So a rerun under a fresh name IS an independent draw.
#
# Describe these as "independent training runs", never as "3 seeds": the seed
# is fixed and saying otherwise would be false. Independent runs is also the
# exact phrase dob9 used in their Q2.
#
# EACH REP TRAINS END TO END, including its own FedAvg trunk. That is
# deliberate: the error bar should cover the whole pipeline's variance, not
# just the fine-tuning step. It does mean these bars are wider than a
# fine-tuning-only spread would be, which is the honest number for a figure
# whose points are whole methods.
#
# Configs copied from the published runs' config.json, not from memory:
#   fedavg      fedavg_full_hilo8_random_nes_save
#   ft_full     fedavg_ft_full_hilo8_random_nes_ftepochs20_adapternone_save
#   ft_lora     fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_save
#   fedlibra    fedavg_ft_E2_R50_..._ft20_irm100w17_nes_z16_xm4_lora_headr8_save
# All four: profile full, R=50, E=2, metrics privacy, no early stopping.
#
# 8 tasks = 4 arms x 2 extra reps. The published run is rep 1, matching how
# varrep was done, so each arm ends with n=3. Each task trains THEN generates
# at 8000/hospital so the scorer has what it needs. Historical wall clock is
# 8-20 min train plus a few min generate; 02:00:00 is margin.
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=reps-fed4
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
BASE="--profile full --cohort-cache $CACHE
      --local-epochs 2 --n-rounds 50 --metrics privacy --no-early-stop"

case "$SLURM_ARRAY_TASK_ID" in
 0) ARM=fedavg;   REP=2 ;;
 1) ARM=fedavg;   REP=3 ;;
 2) ARM=ft_full;  REP=2 ;;
 3) ARM=ft_full;  REP=3 ;;
 4) ARM=ft_lora;  REP=2 ;;
 5) ARM=ft_lora;  REP=3 ;;
 6) ARM=fedlibra; REP=2 ;;
 7) ARM=fedlibra; REP=3 ;;
 *) echo "bad array index" >&2; exit 1 ;;
esac

case "$ARM" in
 fedavg)   ARGS="--regime fedavg" ;;
 ft_full)  ARGS="--regime fedavg_ft --ft-epochs 20 --adapter none --adapter-rank 8" ;;
 ft_lora)  ARGS="--regime fedavg_ft --ft-epochs 20 --adapter lora_head --adapter-rank 8" ;;
 fedlibra) ARGS="--regime fedavg_ft --ft-epochs 20 --irm-rho 100 --irm-warmup 17
                 --latent-dim 16 --xm-k 4 --adapter lora_head --adapter-rank 8" ;;
esac

NAME="rep${REP}_${ARM}"
echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py $BASE $ARGS --run-name "$NAME" --resume
echo "### generating $NAME at 8000/hospital  ($(date +%H:%M:%S))"
python examples/fedpyhealth/generate.py --save-dir "_outputs/${NAME}_save" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000
echo done
