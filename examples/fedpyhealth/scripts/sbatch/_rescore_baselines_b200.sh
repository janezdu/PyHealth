#!/bin/bash
# Rescore the remaining panel arms at 200 resamples, so every point on the
# specialist/generalist figure is measured the same way.
#
# WHY THIS IS NOT OPTIONAL. The same arm scored at n=5 and n=200 moves by up to
# 0.017 on specialist Pearson, against an n=200 standard error of 0.0027 -- six
# standard errors. A panel mixing the two precisions would show differences that
# are an artefact of how many resamples each point happened to get.
#
# No regeneration; every synthetic.json is already on disk.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=rescore-base
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
RES=_outputs/results/tests
H=_outputs/fedavg_ft_full_hilo8_random_nes_adapter

RUNS="--run local=_outputs/local_full_hilo8_random_nes_save
      --run centralized=_outputs/centralized_full_hilo8_random_nes_save
      --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save
      --run fedavg_irm1e2=_outputs/fedavg_full_hilo8_random_irmrho100_save
      --run fedavg_irm1e4=_outputs/fedavg_full_hilo8_random_irmrho10000_save
      --run centralized_irm1e2=_outputs/centralized_full_hilo8_random_irmrho100_save
      --run centralized_irm1e4=_outputs/centralized_full_hilo8_random_irmrho10000_save
      --run lora_attn_ep2=${H}lora_attn_save
      --run lora_head_ep2=${H}lora_head_save
      --run last_mlp_ep2=${H}last_mlp_save"

for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    echo "############ baselines, $tag"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "$RES/test1_baselines_${tag}_b200.json"
done
