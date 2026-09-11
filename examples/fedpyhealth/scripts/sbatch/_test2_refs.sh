#!/bin/bash
# The two reference arms the adapter Test 2 run was missing.
#
# test2_adapters_budget8000.json scored seven FINE-TUNED arms against each other
# and found them indistinguishable. That comparison cannot answer the prior
# question -- whether local fine-tuning moves the utility axis AT ALL -- because
# it contains no un-fine-tuned generator. These two supply it:
#
#   fedavg        the plain federated global, never fine-tuned
#   fedavg_irm100 the same, trained with the rho=1e2 IRM penalty
#
# No regeneration. Both are shared-generator runs, so train.py handed every
# hospital the whole pooled draw (64,000 each); --train-budget 8000 caps that to
# the same classifier volume every other arm gets, which is the control that
# makes the comparison about the generator rather than about record count.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-refs
#SBATCH --time=01:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" \
    --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save \
    --run fedavg_irm100=_outputs/fedavg_full_hilo8_random_irmrho100_save \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_refs_budget8000.json
