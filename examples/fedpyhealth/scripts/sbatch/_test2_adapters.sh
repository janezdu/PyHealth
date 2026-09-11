#!/bin/bash
# Test 2 (downstream utility) across every adapter arm -- the axis the adapter
# work has never been scored on. Test 1 asks "does the synthetic data have the
# right code prevalences"; Test 2 asks "can you train a useful rare-code
# classifier on it", and the two can rank arms differently. An arm that matches
# prevalence by emitting well-calibrated noise scores well on Test 1 and badly
# here, which is the specific thing this catches.
#
# No regeneration: the 8000/hospital synthetic.json from the Test 1 scoring runs
# is already on disk, and re-rolling it would make these numbers incomparable to
# the Test 1 numbers they sit beside.
#
# --train-budget 8000 for the same reason it is used everywhere else: uncapped,
# the shared-generator arms hold 8x what the per-site arms hold and win on
# volume rather than on generator quality.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=test2-adapters
#SBATCH --time=01:30:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

CACHE="$FEDCOHORT_CACHE/hilo8_random"
B=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapter

RUNS=""
add () {  # add <display-name> <save-dir>
    [ -d "$2" ] || { echo "missing $2" >&2; exit 1; }
    [ -f "$2/synthetic.json" ] || { echo "no synthetic.json in $2 -- regenerate first" >&2; exit 1; }
    RUNS="$RUNS --run $1=$2"
}

# The ft_epochs=20 sweep. adapter=none here is the ONLY correct full-fine-tune
# baseline for this set: raising ft_epochs shortens the FedAvg trunk, so a
# 20-epoch arm starts from a 40-round global and a 2-epoch one from 49.
for v in none lora_attn lora_head last_mlp; do
    if [ "$v" = "none" ]; then n=full_ft20; else n="${v}20"; fi
    add "$n" "${B}${v}_save"
done

# The same fine-tuning on top of the rho=1e2 IRM trunk.
for v in none lora_head lora_attn; do
    if [ "$v" = "none" ]; then n=irm_full_ft; else n="irm_${v}"; fi
    add "$n" "${B}${v}_irmrho100_irmwarmup34_save"
done

echo "arms:$RUNS"
echo

python examples/fedpyhealth/test2_rare_efficacy.py \
    --cohort-cache "$CACHE" $RUNS \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_adapters_budget8000.json
