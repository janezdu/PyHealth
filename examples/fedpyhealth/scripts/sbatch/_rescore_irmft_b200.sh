#!/bin/bash
# Table 4's missing cell: IRM + LoRA fine-tuning, WITHOUT best-of-K.
#
# dob9's Q2 asks for exactly this arm by name -- "How does IRM followed by the
# same local LoRA adaptation perform without best-of-K?" -- and all three
# reviewers note that best-of-K alone worsens both axes while the text claims
# otherwise. Without this arm there is no evidence best-of-K contributes
# anything over the simpler IRM + restricted local adaptation.
#
# NOTHING IS TRAINED OR GENERATED HERE. The three runs finished on 31 Aug and
# their synthetic.json files are on disk (7.4-8.5 MB each). _score_irmft.sh
# already scored them -- but at the DEFAULT 5 bootstraps, which is the noise
# floor that made Test 1 look unstable. This rescores the same synthetic at
# 200, matching every other number in the main table.
#
# Arms:
#   irm_full_ft    IRM trunk + full fine-tuning
#   irm_lora_head  IRM trunk + LoRA on the head   <- THE TABLE 4 CELL
#   irm_lora_attn  IRM trunk + LoRA on attention
#
# The LoRA-head variant is the one that matches FEDLIBRA's adapter
# configuration (LoRA head, r=8), so it is the apples-to-apples ablation; the
# other two travel with it for free.
#
# CPU ONLY -- prevalence scoring does no GPU work. CUDA is masked off so this
# cannot silently grab a device.
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=bgyw-delta-cpu
#SBATCH --job-name=rescore-irmft-b200
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=
CACHE="$FEDCOHORT_CACHE/hilo8_random"
B=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapter

RUNS=""
for v in none lora_head lora_attn; do
  d="${B}${v}_irmrho100_irmwarmup34_save"
  [ -f "$d/synthetic.json" ] || { echo "missing $d/synthetic.json" >&2; exit 1; }
  if [ "$v" = "none" ]; then name="irm_full_ft"; else name="irm_${v}"; fi
  RUNS="$RUNS --run ${name}=${d}"
done

echo "############ Test 1, SPECIALIST, 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_irmft_spec_b200.json

echo "############ Test 1, GENERALIST, 200 bootstraps"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out _outputs/results/tests/test1_irmft_gen_b200.json
echo done
