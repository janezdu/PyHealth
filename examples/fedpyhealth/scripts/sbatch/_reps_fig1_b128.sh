#!/bin/bash
# Error bars for Figure 1, at the batch size the figure was actually produced
# at.
#
# Five of the six published Figure 1 arms were trained at batch 128; FedLIBRA
# alone was trained at 256 (sweeps/b128.yaml's own header says 128 was "the
# standing choice for every cohort", so 256 is the anomaly). The earlier
# replicate array ran everything at 256, which gave tight bars around the
# WRONG points and is what surfaced the batch-size difference in the first
# place.
#
# This runs all six at 128, two independent runs each. Combined with the
# published run, five arms reach n=3 at their own published setting. FedLIBRA
# gets its first 128 runs at all -- the control for "does batch 256 lift
# everything, or only the fine-tuning-heavy arms?", which is currently
# unanswered because no FedLIBRA run at 128 exists.
#
# PER-ARM SETTINGS COPIED FROM EACH PUBLISHED RUN'S config.json. Note
# centralized is the odd one: it is the only arm published WITH rare-code
# upweighting and WITH early stopping, so it does not take --no-early-stop.
# Getting that wrong would put its bar on a different point than the figure.
#
# These are independent training runs, not seeds: SEED stays 0 at train.py:74
# and the variation comes from training nondeterminism. Do not call them seeds
# in the paper.
#
# 128 doubles the optimizer steps against the 256 runs, which took 8-23 min,
# so budget 1.5-2x: ~40 min a task. 01:30:00 is 2x headroom and backfills far
# better than a 3h request on a partition with ~2,400 jobs pending. --resume
# means a task that somehow hit the wall can continue rather than restart.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=reps-fig1-b128
#SBATCH --time=01:30:00
#SBATCH --array=0-11
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
# b128.yaml sets profile: full and batch_size: 128.
BASE="--config examples/fedpyhealth/sweeps/b128.yaml --cohort-cache $CACHE
      --local-epochs 2 --n-rounds 50 --metrics privacy"

ARMS=(local centralized fedavg ft_full ft_lora fedlibra)
# Default: --array=0-11, two reps per arm, numbered 2 and 3.
# REP_FIXED=<n> with --array=0-5 instead does ONE rep per arm, numbered <n>.
# That is how the third run of each arm is added without touching this file:
#   sbatch --array=0-5 --export=ALL,REP_FIXED=4 <this script>
if [ -n "${REP_FIXED:-}" ]; then
    ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
    REP=$REP_FIXED
else
    ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 2))]}
    REP=$((SLURM_ARRAY_TASK_ID % 2 + 2))
fi

case "$ARM" in
 local)       ARGS="--regime local --no-early-stop" ;;
 # the ONLY arm published upweighted and early-stopped -- no --no-early-stop
 centralized) ARGS="--regime centralized --rare-upweight" ;;
 fedavg)      ARGS="--regime fedavg --no-early-stop" ;;
 ft_full)     ARGS="--regime fedavg_ft --no-early-stop --ft-epochs 20
                    --adapter none --adapter-rank 8" ;;
 ft_lora)     ARGS="--regime fedavg_ft --no-early-stop --ft-epochs 20
                    --adapter lora_head --adapter-rank 8" ;;
 fedlibra)    ARGS="--regime fedavg_ft --no-early-stop --ft-epochs 20
                    --irm-rho 100 --irm-warmup 17 --latent-dim 16 --xm-k 4
                    --adapter lora_head --adapter-rank 8" ;;
 *) echo "bad arm $ARM" >&2; exit 1 ;;
esac

NAME="b128rep${REP}_${ARM}"
echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py $BASE $ARGS --run-name "$NAME" --resume
echo "### generating $NAME at 8000/hospital  ($(date +%H:%M:%S))"
python examples/fedpyhealth/generate.py --save-dir "_outputs/${NAME}_save" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000
echo done
