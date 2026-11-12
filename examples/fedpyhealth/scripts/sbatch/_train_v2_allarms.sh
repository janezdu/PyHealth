#!/bin/bash
# v2: every arm on the frontier figure, retrained on ONE declared budget.
#
# WHY. The current figure's arms are budget-matched at 100 passes/hospital
# (n_rounds 50 x local_epochs 2) with two exceptions that this rerun fixes:
#
#   centralized_rw  early-stopped after 17 of 100 epochs, and it is the FRONTIER
#                   ANCHOR -- every "distance above the line" was measured from
#                   a point that got one-sixth the compute. Here it keeps
#                   --rare-upweight and gains --no-early-stop.
#   fedavg_ft       the plotted arm is ft_epochs=2 while the legend says 20.
#                   Both are run here so the label can stop guessing.
#
# Budget: 100 passes/hospital for EVERY arm. train.py splits rather than adds
# for fine-tuning regimes -- fed_rounds = (100 - ft) // 2 -- so ft2 spends
# 49 rounds + 2 and ft20 spends 40 rounds + 20. Both total 100.
#
# Names are prefixed v2_ so nothing here collides with the existing runs, which
# stay on disk for comparison.
#
# Array, not one sequential job: gpuA40x4 has ~1670 pending and backfills short
# jobs, so nine 2h tasks start far sooner than one 9h job.
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=v2-arms
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
BASE="--profile full --cohort-cache $FEDCOHORT_CACHE/hilo8_random
      --local-epochs 2 --n-rounds 50 --metrics privacy --no-early-stop"

case "$SLURM_ARRAY_TASK_ID" in
 0) NAME=v2_local;              ARGS="--regime local" ;;
 1) NAME=v2_fedavg;             ARGS="--regime fedavg" ;;
 2) NAME=v2_centralized_rw;     ARGS="--regime centralized --rare-upweight" ;;
 3) NAME=v2_fedavg_ft2;         ARGS="--regime fedavg_ft --ft-epochs 2" ;;
 4) NAME=v2_fedavg_ft20;        ARGS="--regime fedavg_ft --ft-epochs 20" ;;
 5) NAME=v2_fedavg_irm1e4;      ARGS="--regime fedavg --irm-rho 10000 --irm-warmup 17" ;;
 6) NAME=v2_fedavg_xm4;         ARGS="--regime fedavg --dropout 0.3 --xm-k 4" ;;
 7) NAME=v2_fedavg_irm_xm4;     ARGS="--regime fedavg --irm-rho 100 --irm-warmup 17 --dropout 0.1 --xm-k 4" ;;
 8) NAME=v2_fedavg_irm_xm4_ft20;
    ARGS="--regime fedavg_ft --ft-epochs 20 --irm-rho 100 --irm-warmup 17
          --latent-dim 16 --xm-k 4 --adapter lora_head --adapter-rank 8" ;;
 # 9-10: the two missing cells of the upweighting x early-stopping 2x2.
 # Appendix "Rare-code upweighting" compares unweighted against upweighted for
 # local-only, FedAvg and centralized -- but the upweighted runs early-stopped
 # and the unweighted ones ran the full budget, so weighting and stopping move
 # together in every row. Its own Caveat paragraph says so. Task 2
 # (v2_centralized_rw) supplies the centralized cell; these two complete the
 # other rows, so the caveat can be deleted rather than narrowed.
 #
 # BASE already carries --no-early-stop, so each of these is the upweighted,
 # full-budget cell its row is missing. Run as: sbatch --array=2,9,10
 9) NAME=v2_local_rw;           ARGS="--regime local --rare-upweight" ;;
10) NAME=v2_fedavg_rw;          ARGS="--regime fedavg --rare-upweight" ;;
 *) echo "bad array index" >&2; exit 1 ;;
esac

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py $BASE $ARGS --run-name "$NAME" --resume
echo done
