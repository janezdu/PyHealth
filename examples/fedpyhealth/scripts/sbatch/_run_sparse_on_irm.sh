#!/bin/bash
# The missing cell: sparse fine-tuning on top of the IRM trunk.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/_run_sparse_on_irm.sh
#
# We have plain-trunk x {full FT, LoRA head, LoRA attn, IHT, L1} and
# IRM-trunk x {full FT, LoRA head, LoRA attn}. The sparse arms on an IRM trunk
# were never run, and that is the cell worth knowing: irm_lora_head is currently
# the best arm on the board (0.633 specialist / 0.725 generalist R^2), and both
# IRM and sparsity act on ABSOLUTE calibration rather than code ranking -- so
# they may compose, or may turn out to be two routes to the same correction.
#
# THE TRUNK. Seeded from fedavg_full_hilo8_random_irmrho100_save (50 rounds,
# rho=1e2) -- the SAME checkpoint the existing irm_* arms used, so these are
# directly comparable to them. fed_rounds at ft_epochs=20 is 40, so --resume
# sees 50 >= 40 and skips straight to fine-tuning.
#
# --irm-rho/--irm-warmup are passed for the RUN NAME and trunk identity only.
# The IRM penalty never applies during fine-tuning (train.py forces irm_rho=0
# there) because IRM enforces cross-site invariance and local fine-tuning
# deliberately breaks it. So this is "IRM trunk, sparse local fine-tuning".
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sparse-on-irm
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

CACHE="$FEDCOHORT_CACHE/hilo8_random"
TRUNK=_outputs/fedavg_full_hilo8_random_irmrho100_save/fedavg_state.pt
[ -f "$TRUNK" ] || { echo "no IRM trunk at $TRUNK" >&2; exit 1; }

COMMON="--profile full --regime fedavg_ft --cohort-cache $CACHE \
        --no-early-stop --ft-epochs 20 --resume \
        --irm-rho 100 --irm-warmup 34"

declare -a ARMS=(
  "irm_iht50:--adapter last_mlp_iht --adapter-sparsity 0.5"
  "irm_iht10:--adapter last_mlp_iht --adapter-sparsity 0.1"
  "irm_iht07:--adapter last_mlp_iht --adapter-sparsity 0.07"
  "irm_l1_02:--adapter last_mlp_l1 --adapter-l1 0.2"
  "irm_l1_03:--adapter last_mlp_l1 --adapter-l1 0.3"
  "irm_l1_05:--adapter last_mlp_l1 --adapter-l1 0.5"
)

RUNS=""
for entry in "${ARMS[@]}"; do
    name="${entry%%:*}"; flags="${entry#*:}"
    dir=$(python examples/fedpyhealth/train.py $COMMON $flags --print-save-dir | tail -1)
    if [ -z "$dir" ] || [ "${dir#_outputs/}" = "$dir" ]; then
        echo "could not resolve a save_dir for $name (got '$dir')" >&2; exit 1
    fi
    mkdir -p "$dir"
    [ -f "$dir/fedavg_state.pt" ] || cp "$TRUNK" "$dir/fedavg_state.pt"
    echo "############ training $name -> $dir"
    python examples/fedpyhealth/train.py $COMMON $flags
    echo "############ regenerating $name at 8000/hospital"
    python examples/fedpyhealth/generate.py --save-dir "$dir" \
        --cohort-cache "$CACHE" --synth-per-hospital 8000
    RUNS="$RUNS --run ${name}=${dir}"
done

# Scored here at n=200 with the shared seed so these land directly comparable to
# every other _b200 file. Test 1 is CPU work, but it is a couple of minutes on
# the end of a job that already holds the node.
for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    echo "############ Test 1, $tag"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "_outputs/results/tests/test1_sparse_on_irm_${tag}_b200.json"
done
