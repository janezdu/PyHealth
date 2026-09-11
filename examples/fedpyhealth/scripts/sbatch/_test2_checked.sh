#!/bin/bash
# Test 2 (downstream rare-code utility, TSTR) on the four families currently
# checked on the specialist/generalist panel: no fine-tuning, IRM trunk +
# LoRA/full, sparse L1 on the plain trunk, and best-of-K + dropout 0.3.
#
# No regeneration -- every arm's synthetic.json is already at 8000/hospital from
# its Test 1 scoring, and re-rolling it would make these numbers incomparable to
# the Test 1 numbers they sit beside.
#
# --train-budget 8000 for the usual reason: uncapped, the shared-generator arms
# hold 8x what the per-site arms hold and would win on volume rather than on
# generator quality.
#
# Worth remembering what Test 2 has said so far: it could not separate ANY
# fine-tuned arm from the un-fine-tuned global (0.0089-0.0097 overall AP against
# fedavg's 0.0096), and its measurement noise is only +/-0.0002. If the XM arms
# move it, that would be the first thing in this project to do so.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-checked
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" \
    --run local=_outputs/local_full_hilo8_random_nes_save --run centralized=_outputs/centralized_full_hilo8_random_nes_save --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save --run trunk=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_nes_last_mlp_l1r8_l11_save --run irm_full_ft=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapternone_irmrho100_irmwarmup34_save --run irm_lora_attn=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_attn_irmrho100_irmwarmup34_save --run irm_lora_head=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_irmrho100_irmwarmup34_save --run l1_02=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_nes_last_mlp_l1r8_l10.2_save --run l1_03=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_nes_last_mlp_l1r8_l10.3_save --run l1_05=_outputs/fedavg_ft_E2_R50_hilo8_random_privacy_ft20_nes_last_mlp_l1r8_l10.5_save --run xm2=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm2_save --run xm4=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm4_save --run xm8=_outputs/centralized_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save --run xm8_fedavg=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_do0.3_xm8_save \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_checked_budget8000.json
