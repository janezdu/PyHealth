#!/bin/bash
# Rescore Test 1 at high precision. CPU ONLY -- test1_prevalence.py is pandas +
# sklearn and contains no CUDA reference at all, so a GPU allocation for it is
# pure waste (an A40 job bills 500/hr to compute correlations).
#
# Delta has no *-delta-cpu account here and rejects zero-GPU jobs under the gpu
# accounts, so the sane place for this is an interactive node you already hold,
# or a login shell if it is short:
#
#   srun --account=bgyw-delta-gpu --partition=gpuA40x4 --gpus-per-node=1 \
#        --cpus-per-task=8 --mem=32g --time=01:00:00 --pty bash
#   source .venv/bin/activate
#   bash examples/fedpyhealth/scripts/rescore_cpu.sh
#
# No regeneration -- every synthetic.json is already on disk; this only re-runs
# the metric over it.
#
# WHY n=200. Scored at the old default of 5, the arm ranking is not stable:
# full_ft20 and lora_head20 swap first place on Pearson between n=5 and n=200,
# and last_mlp20/lora_head20 swap on R^2. The reported value is the MEAN of n
# resamples over codes, so its standard error falls as 1/sqrt(n) -- 0.098 at
# n=5, 0.018 at n=200 for specialist R^2.
set -euo pipefail

CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"
RES=_outputs/results/tests
H=_outputs/fedavg_ft_full_hilo8_random_nes_adapter
NB="${NB:-200}"
SEED="${SEED:-0}"

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
    out="$RES/test1_baselines_${tag}_b${NB}.json"
    if [ -f "$out" ]; then
        echo "skip $out (already there; delete it to redo)"
        continue
    fi
    echo "############ baselines, $tag  (n=$NB, seed=$SEED)"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps "$NB" --bootstrap-seed "$SEED" --out "$out"
done
echo
echo "Done. The panel needs test1_baselines_{spec,gen}_b200.json plus the"
echo "adapters20 and sparse _b200 files, which already exist."
