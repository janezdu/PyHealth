#!/bin/bash
# Smoke for --regime perada_fl: smallest budget that exercises every stage.
#
#   tiny model, 3 passes/hospital = 1 trunk round (1 local epoch) + 1 PerAda
#   round (1 epoch local adapter + 1 epoch personal adapter), KD pool 16, 2 KD
#   steps. Then generate.py against the save_dir.
#
# WHAT TO READ IN THE LOG, since none of these fail loudly on their own:
#   "PerAda round 1/1: mean ||v_m - w|| = X"   X must be > 0, or the personal
#                                              adapter never trained
#   "KD a -> b"                                both finite
#   eight "PerAda hospital <hid> -> .../ft_<hid>.pt" lines
#   generate.py finishing with 8 hospitals     proves the adapter-wrapped
#                                              checkpoints load back
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=perada-fl-smoke
#SBATCH --time=00:30:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"

ARGS="--profile tiny --cohort-cache $CACHE --no-tb --no-early-stop \
      --regime perada_fl --n-rounds 3 --local-epochs 1 --ft-epochs 2 \
      --adapter lora_head --adapter-rank 8 \
      --perada-kd-pool 16 --perada-kd-steps 2 --perada-kd-batch 8"

D=$(python examples/fedpyhealth/train.py $ARGS --print-save-dir | tail -1)
echo "### save_dir = $D"
case "$D" in
    _outputs/?*) rm -rf "$D" ;;   # a smoke must start clean, never resume one
    *) echo "refusing to delete unexpected save_dir '$D'" >&2; exit 1 ;;
esac

python examples/fedpyhealth/train.py $ARGS

echo "### ft checkpoints:"; ls "$D"/ft_*.pt
echo "### generate.py round-trip"
python examples/fedpyhealth/generate.py --save-dir "$D" \
    --cohort-cache "$CACHE" --synth-per-hospital 16
echo SMOKE_OK
