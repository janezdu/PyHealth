#!/bin/bash
# Fill the two holes in the best-of-K families so the dropout and latent
# versions of the frontier figure are symmetric.
#
# The figure wants two matched pairs -- (best-of-K) and (best-of-K + fine-tune)
# -- once with dropout stochasticity and once with latent. Today:
#
#   latent  z16+IRM+K4          scored      <- filled green
#   latent  z16+IRM+K4 + FT     scored      <- hollow green
#   dropout do0.1+IRM+K4        scored      <- filled purple
#   dropout do0.1+IRM+K4 + FT   DOES NOT EXIST   <- the missing hollow purple
#   latent  z16+K4, no IRM      trained, NEVER SCORED
#
# Task 1 trains the missing dropout fine-tune arm -- identical to the latent
# hollow-star run in every respect (ft20, LoRA head r8, IRM 100/w17, K=4) except
# that its K draws differ by dropout 0.1 instead of a z=16 latent. That one
# substitution is the whole point: with it, "what does fine-tuning buy" can be
# asked inside BOTH families instead of only one.
#
# Task 2 scores lat_xm4 (trained yesterday, job 22187837), the no-IRM latent
# arm. It is the latent twin of the orange square, and the IRM ablation inside
# the latent family.
#
# Both scored into ONE file at 200 bootstraps, matching the settings every
# other b200 file on the figure uses.
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=xm-fill
#SBATCH --time=03:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
BASE="--profile full --cohort-cache $CACHE --local-epochs 2 --n-rounds 50
      --metrics privacy --no-early-stop"

# --- 1. the missing dropout fine-tune arm ---------------------------------
echo "############ training do_lora_ft (the missing hollow purple star)"
python examples/fedpyhealth/train.py $BASE --regime fedavg_ft \
    --ft-epochs 20 --irm-rho 100 --irm-warmup 17 --dropout 0.1 --xm-k 4 \
    --adapter lora_head --adapter-rank 8 \
    --run-name do_lora_ft --resume

D_DO=_outputs/do_lora_ft_save
D_LAT=_outputs/fedavg_E2_R50_hilo8_random_privacy_nes_z16_xm4_save

# --- 2. generate + score both ---------------------------------------------
RUNS=""
for pair in "do_lora_ft=$D_DO" "lat_xm4=$D_LAT"; do
    name="${pair%%=*}"; dir="${pair#*=}"
    [ -d "$dir" ] || { echo "missing $dir" >&2; exit 1; }
    if [ ! -f "$dir/synthetic.json" ]; then
        echo "############ generating for $name"
        python examples/fedpyhealth/generate.py --save-dir "$dir" \
            --cohort-cache "$CACHE" --synth-per-hospital 8000
    fi
    RUNS="$RUNS --run $name=$dir"
done

# hospital scope -> the specialist axis; pooled -> the generalist axis.
for pair in "hospital:spec" "pooled:gen"; do
    scope="${pair%%:*}"; tag="${pair##*:}"
    echo "############ scoring $tag (real-scope $scope)"
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 8000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "_outputs/results/tests/test1_xmfill_${tag}_b200.json"
done
echo done
