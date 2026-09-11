#!/bin/bash
# Score "IRM trunk + local fine-tuning": the composition the adapter design was
# aimed at. The FedAvg half is NOT retrained -- each run's save_dir was seeded
# with the exact fedavg_state.pt from fedavg_full_hilo8_random_irmrho100_save
# (50 rounds, rho=1e2), and fed_rounds at ft_epochs=20 is 40, so run_fedavg sees
# 50 >= 40 and skips straight to fine-tuning. That makes these directly
# comparable to the IRM point already on the plot: same trunk, plus an adapter.
set -euo pipefail
CACHE="$FEDCOHORT_CACHE/hilo8_random"
B=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapter
RUNS=""
for v in none lora_head lora_attn; do
  d="${B}${v}_irmrho100_irmwarmup34_save"
  [ -d "$d" ] || { echo "missing $d" >&2; exit 1; }
  echo "############ regenerating irm+$v at 8000/hospital"
  python examples/fedpyhealth/generate.py --save-dir "$d" \
      --cohort-cache "$CACHE" --synth-per-hospital 8000
  if [ "$v" = "none" ]; then name="irm_full_ft"; else name="irm_${v}"; fi
  RUNS="$RUNS --run ${name}=${d}"
done
echo "############ Test 1, SPECIALIST"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --out _outputs/results/tests/test1_irmft_spec.json
echo "############ Test 1, GENERALIST"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --out _outputs/results/tests/test1_irmft_gen.json
