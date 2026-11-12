#!/bin/bash
# Smoke the five literature baselines on --profile tiny. Each is a full run
# end to end -- train, generate, score -- so a crash anywhere in the wiring
# shows up here rather than 25 minutes into a real job.
#
#   srun --account=bgyw-delta-gpu --partition=gpuA40x4 --gpus-per-node=1 \
#        --cpus-per-task=8 --mem=32g --time=01:00:00 --pty bash
#   source .venv/bin/activate
#   bash examples/fedpyhealth/scripts/smoke_fed_baselines.sh
#
# WHAT EACH ONE IS CHECKING, beyond "it runs":
#   fedprox    the round log prints a prox mu and the run still ends with ONE
#              global model
#   fedrep     "personal_mode=head keeps N/M params" appears, and eight
#              ft_<hid>.pt files are written without a fine-tuning phase
#   lg_fedavg  same, with the complementary split
#   ditto      each site logs ||v - w*||; at a large lambda that number is small
#   perada     "PerAda: KD toward the frozen global" appears
set -euo pipefail
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"

run () {
    echo; echo "############ $1"
    shift
    python examples/fedpyhealth/train.py --profile tiny --cohort-cache "$CACHE" \
        --no-tb "$@" 2>&1 | tail -25
}

run "fedprox"   --regime fedprox   --fedprox-mu 0.01
run "fedrep"    --regime fedrep
run "lg_fedavg" --regime lg_fedavg
run "ditto"     --regime ditto     --ditto-lambda 1.0
run "perada"    --regime perada    --adapter lora_head --kd-lambda 1.0
echo; echo "all five completed"
