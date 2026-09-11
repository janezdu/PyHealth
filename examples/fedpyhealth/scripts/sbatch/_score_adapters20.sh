#!/bin/bash
# Score the 20-epoch fine-tuning sweep. Separate baseline from the 2-epoch runs:
# raising ft_epochs shortens the FedAvg trunk (fed_rounds = (100 - ft_epochs)/2),
# so a 20-epoch adapter starts from a 40-round global while a 2-epoch one starts
# from 49. The adapter=none run at ft_epochs=20 is therefore the ONLY correct
# full-fine-tune baseline for this set.
set -euo pipefail
CACHE="$FEDCOHORT_CACHE/hilo8_random"
B=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapter
RUNS=""
for v in none lora_attn lora_head last_mlp; do
  d="${B}${v}_save"
  if [ ! -d "$d" ]; then echo "missing $d" >&2; exit 1; fi
  echo "############ regenerating $v at 8000/hospital"
  python examples/fedpyhealth/generate.py --save-dir "$d" \
      --cohort-cache "$CACHE" --synth-per-hospital 8000
  if [ "$v" = "none" ]; then name="full_ft20"; else name="${v}20"; fi
  RUNS="$RUNS --run ${name}=${d}"
done
echo "############ Test 1, SPECIALIST"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --out _outputs/results/tests/test1_adapters20_spec.json
echo "############ Test 1, GENERALIST"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --out _outputs/results/tests/test1_adapters20_gen.json
