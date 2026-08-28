#!/bin/bash
# Validate the adapter implementation before spending GPU hours on it.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/smoke_adapters.sbatch
#
# Three of the four properties these adapters need fail SILENTLY:
#
#   * a LoRA that is not zero-initialised changes the model before training, so
#     "adapter = 0 means be the federated global" stops being true and the
#     FedProx term is measuring the wrong thing;
#   * a trunk that is not actually frozen makes the run a full fine-tune wearing
#     an adapter's name, with a parameter count that is simply false;
#   * a LoRA on the autoregressive head that does not respect the causal mask
#     lets a code see codes that follow it in the same visit -- and training loss
#     IMPROVES when this breaks, so the tell points the wrong way.
#
# tests/core/test_halo_adapters.py covers all three. This runs them on a compute
# node (importing torch on the login node takes minutes) and then exercises each
# variant end to end on the tiny profile, because the unit tests bind
# _irm_terms-style helpers directly and cannot catch a break in how train.py
# calls them.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-adapters
#SBATCH --time=00:40:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

CACHE="$FEDCOHORT_CACHE/hilo8_random"

echo "############ 1/2  unit tests"
python -m unittest tests.core.test_halo_adapters -v

echo
echo "############ 2/2  tiny end-to-end, every variant"
# fedavg_ft is the only regime the adapter touches: it is the one with a local
# fine-tuning stage. mu is exercised on one variant so the proximal path is not
# first run at full scale.
for a in lora_attn last_mlp lora_head; do
    echo "---- $a"
    python examples/fedpyhealth/train.py --profile tiny --regime fedavg_ft \
        --cohort-cache "$CACHE" --adapter "$a" --adapter-rank 8 \
        --run-name "smoke_adapter_${a}" --no-tb
done

echo "---- lora_attn with mu"
python examples/fedpyhealth/train.py --profile tiny --regime fedavg_ft \
    --cohort-cache "$CACHE" --adapter lora_attn --adapter-rank 8 \
    --adapter-mu 0.01 --run-name smoke_adapter_mu --no-tb

echo
echo "All variants ran. Check the per-hospital 'training N of M params' lines"
echo "above: lora_attn ~0.5%, lora_head ~0.6%, last_mlp ~8.5% of the trunk."
