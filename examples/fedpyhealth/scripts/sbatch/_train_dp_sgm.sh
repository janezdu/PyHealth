#!/bin/bash
# Privacy runs, all FedAvg on hilo8_random, 10 rounds x 10 local epochs, uniform
# weighting, clip 1.0, delta 1e-5.
#
#   0-2 Gaussian DP-FedAvg REPEATED from scratch. The earlier dp_s* runs noised
#       HALO's float attention mask (see dp_aggregate), so they are invalid.
#   3-5 Fed-SGM on lora_head r8 lr 1e-3, trunk frozen at pub8, b=40,000.
#   6-8 Fed-SGM on all parameters, warm-started from pub8, b=100,000.
#
# sigma_g is from dpaccount.sgm_sigma_for at epsilon 9, 20, 50. Submit 0-5 with
# --time=01:00:00 and 6-8 with --time=04:00:00: the full sketch applies a
# 100,000 x 6.2M matrix to every client every round.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=dp-sgm
#SBATCH --output=_outputs/slurm/%x-%A_%a.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
: "${INIT_FROM:=_outputs/centralized_full_pub8_nrounds25_localepochs2_save}"

python tests/core/test_dp_sgm.py

INIT="--init-from $INIT_FROM"
HEAD="--adapter lora_head --adapter-rank 8 --adapter-lr 1e-3"
case $SLURM_ARRAY_TASK_ID in
 0) NAME=dpfix_s0.5;    ARGS="--dp-sigma 0.5" ;;
 1) NAME=dpfix_s1.0;    ARGS="--dp-sigma 1.0" ;;
 2) NAME=dpfix_s2.0;    ARGS="--dp-sigma 2.0" ;;
 3) NAME=sgm_head_e9;   ARGS="$INIT $HEAD --dp-mech sgm --dp-sketch-b 40000 --dp-sigma 0.1261" ;;
 4) NAME=sgm_head_e20;  ARGS="$INIT $HEAD --dp-mech sgm --dp-sketch-b 40000 --dp-sigma 0.0942" ;;
 5) NAME=sgm_head_e50;  ARGS="$INIT $HEAD --dp-mech sgm --dp-sketch-b 40000 --dp-sigma 0.0704" ;;
 6) NAME=sgm_full_e9;   ARGS="$INIT --dp-mech sgm --dp-sketch-b 100000 --dp-sigma 0.1003" ;;
 7) NAME=sgm_full_e20;  ARGS="$INIT --dp-mech sgm --dp-sketch-b 100000 --dp-sigma 0.0749" ;;
 8) NAME=sgm_full_e50;  ARGS="$INIT --dp-mech sgm --dp-sketch-b 100000 --dp-sigma 0.0560" ;;
esac

echo "### $NAME  ($(date +%H:%M:%S))"
python examples/fedpyhealth/train.py --profile full \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --regime fedavg --n-rounds 10 --local-epochs 10 --metrics privacy \
    --no-early-stop --weighting uniform --dp-clip 1.0 \
    $ARGS --run-name "$NAME" --resume
echo done
