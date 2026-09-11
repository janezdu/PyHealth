#!/bin/bash
# Regenerate and score the six adapter runs against the full fine-tune baseline.
# Called by the fix-adapters job; separate so the run list lives in a file rather
# than inside an --wrap string that cannot be read back later.
set -euo pipefail
CACHE="$FEDCOHORT_CACHE/hilo8_random"
BASE=_outputs/fedavg_ft_full_hilo8_random_nes_adapter
RUNS="--run full_ft=_outputs/fedavg_ft_full_hilo8_random_nes_save"

for v in lora_attn lora_head last_mlp; do
  for s in "" "_adaptermu0.01"; do
    d="${BASE}${v}${s}_save"
    # if/else, not `[ -n "$s" ] && ...`: under `set -e` a false test makes the
    # assignment exit non-zero and kills the script on the first iteration.
    if [ -n "$s" ]; then name="${v}_mu"; else name="$v"; fi
    echo "############ regenerating $name at 8000/hospital"
    python examples/fedpyhealth/generate.py --save-dir "$d" \
        --cohort-cache "$CACHE" --synth-per-hospital 8000
    RUNS="$RUNS --run ${name}=${d}"
  done
done

echo "############ Test 1, SPECIALIST (each site vs its own fold)"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --out _outputs/results/tests/test1_adapters_spec.json

echo "############ Test 1, GENERALIST (all sites vs the pooled fold)"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --out _outputs/results/tests/test1_adapters_gen.json
