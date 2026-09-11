#!/bin/bash
# IRM sweep: rho below the current best, and four ways of getting there.
#
#   bash examples/fedpyhealth/scripts/sbatch/_irm_sweep.sh          # submit
#   bash examples/fedpyhealth/scripts/sbatch/_irm_sweep.sh --dry-run
#
# Submits a fast unit-test job first and makes every run depend on it, so a
# broken schedule fails in two minutes rather than after seven training runs.
#
# BASELINE TO MATCH. The current best IRM arm is
# fedavg_full_hilo8_random_irmrho100_save: regime fedavg, 50 rounds x 2 local
# epochs, rho=1e2, warmup=17, early stop off. Every run below matches it and
# varies exactly one thing, so nothing here needs a caveat about differing
# budgets. Note warmup=17, not 34 -- the 34 in some run NAMES came from the
# fine-tuning arms and refers to the same trunk.
#
# WHAT VARIES
#   rho      1 and 10, filling in below the best (we have 1e2, 1e4, 1e5)
#   warmup   0 and 34 (we have 17 and 68). Late start already recovered less
#            than half the gap, so the question is whether EARLIER is better
#            still or whether 17 is already past the useful point.
#   schedule geom / cosine / decay at the best rho and warmup. 'decay' is the
#            falsification arm: if starting hot and cooling to 1 matches
#            always-on, the penalty only shapes the early representation and
#            every epoch after ~60 is paying second-order backward for nothing.
#
# ~12 min each on one A100 (measured: 21509426 took 13:51, 21509429 11:53),
# submitted in parallel.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY=1

CACHE="${FEDCOHORT_CACHE:-_outputs/cache/fedcohort}/hilo8_random"
COMMON="--profile full --regime fedavg --cohort-cache $CACHE"

# name : extra flags
declare -a RUNS=(
  "rho1:--irm-rho 1 --irm-warmup 17"
  "rho10:--irm-rho 10 --irm-warmup 17"
  "warm0:--irm-rho 100 --irm-warmup 0"
  "warm34:--irm-rho 100 --irm-warmup 34"
  "geom:--irm-rho 100 --irm-warmup 17 --irm-schedule geom"
  "cosine:--irm-rho 100 --irm-warmup 17 --irm-schedule cosine"
  "decay:--irm-rho 100 --irm-warmup 17 --irm-schedule decay"
)

mkdir -p _outputs/slurm
GATE=_outputs/slurm/launch/_irm_gate.sbatch
mkdir -p "$(dirname "$GATE")"
cat > "$GATE" <<'GEOF'
#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=irm-gate
#SBATCH --time=00:20:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
python -m unittest tests.core.test_halo_irm -v 2>&1 | tail -40
GEOF

if [ -n "$DRY" ]; then
    echo "would submit gate: $GATE"
    for r in "${RUNS[@]}"; do
        echo "would submit ${r%%:*}  ->  python examples/fedpyhealth/train.py $COMMON ${r#*:}"
    done
    exit 0
fi

GID=$(sbatch --parsable "$GATE")
echo "gate job $GID  (unit tests; everything below waits on it)"

for entry in "${RUNS[@]}"; do
    name="${entry%%:*}"; flags="${entry#*:}"
    f="_outputs/slurm/launch/_irm_${name}.sbatch"
    cat > "$f" <<SEOF
#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=irm-${name}
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
python examples/fedpyhealth/train.py $COMMON $flags
SEOF
    sync
    jid=$(sbatch --parsable --dependency=afterok:$GID "$f")
    echo "  $jid  irm-${name}   $flags"
done
echo
echo "When they finish, score them together at n=200 on an interactive node:"
echo "  see examples/fedpyhealth/scripts/score_irm_sweep.sh (written alongside)"
