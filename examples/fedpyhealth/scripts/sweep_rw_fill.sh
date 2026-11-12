#!/bin/bash
# Fill the upweighted (rwfig) frontier with the arms it is missing:
#   lora_head20  FedAvg + LoRA-head fine-tune, 20 ep
#   full_ft20    FedAvg + full fine-tune, 20 ep (rwfig only has a 2-ep one)
#   fedlibra     FedAvg + IRM(100, warmup 17) + best-of-K 4 (z=16) + LoRA head r8
# Same as sweep_rw.sh: --rare-upweight, early stopping at train.py's default
# (ON). Each arm trains its own trunk (no shared-trunk --resume).
#
# Appends to the existing rwfig manifest. Run from the repo root (login node is
# fine, it only submits):
#   bash examples/fedpyhealth/scripts/sweep_rw_fill.sh
set -euo pipefail
MANIFEST="_outputs/results/rwfig_manifest.txt"
mkdir -p _outputs/slurm

sub () {
    local arm="$1"; local regime="$2"; shift 2
    local id
    id=$(sbatch --parsable --job-name="rw-$arm" --time=12:00:00 \
         --export=ALL,ARM="$arm",ARGS="$*",MANIFEST="$MANIFEST",REGIME="$regime",ES_FLAG= \
         examples/fedpyhealth/scripts/sbatch/_sweep_arm.sh)
    echo "  $id  $arm  ($regime)  <- $*"
}

sub lora_head20 fedavg_ft --rare-upweight --ft-epochs 20 --adapter lora_head
sub full_ft20   fedavg_ft --rare-upweight --ft-epochs 20
sub fedlibra    fedavg_ft --rare-upweight --ft-epochs 20 --adapter lora_head \
    --adapter-rank 8 --irm-rho 100 --irm-warmup 17 --xm-k 4 --latent-dim 16
