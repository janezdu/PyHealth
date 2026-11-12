#!/bin/bash
# Figure 1, every arm in ONE setting: batch 128, rare-code upweighting ON,
# early stopping ON. Three independent runs each. Trains, generates and
# scores. No SLURM -- for a workstation with a GPU and no queue.
#
# WHY UNIFORM. The published figure gives different arms different treatments:
# centralized alone is upweighted and early-stopped, and FedLIBRA alone was
# trained at batch 256 while the other five ran at 128. Both asymmetries point
# the same way. Zda4's Q2 asks for exactly this fix -- "Why is upweighting used
# only for the pooled reference? Please retrain all methods with it."
#
# WHY UPWEIGHTED RATHER THAN PLAIN. Unweighted centralized collapses to
# R^2 ~ -45, which makes the frontier's right-hand anchor meaningless. The
# measured 2x2 puts upweighting at about +45.7 on that arm against +7.5 for
# early stopping, so upweighting is what rescues it. Giving every arm both is
# the only matched setting in which the anchor is sane.
#
# Expect upweighting to HURT local-only (measured: 0.759 -> 0.581 specialist)
# and help the federated arms. That is a real result, not a bug -- the appendix
# already says upweighting does not help every approach.
#
# THESE ARE INDEPENDENT TRAINING RUNS, NOT SEEDS. train.py hardcodes SEED = 0
# and has no --seed flag; a rerun under a new --run-name is still an
# independent draw because training is nondeterministic on GPU. Never call
# them seeds in the paper.
#
# ~15 min a run on a single modern GPU, so ~4.5 h for 18, plus a few minutes to
# score. --resume continues an interrupted run and finished runs are skipped,
# so re-running this script is safe and cheap.
#
#   export FEDCOHORT_CACHE=$HOME/cache/fedcohort      # holds hilo8_random/
#   bash examples/fedpyhealth/scripts/run_fig1_uniform.sh
#
# Subsets, for a smoke test before committing the night:
#   REPS=1 ARMS="fedavg fedlibra" bash examples/fedpyhealth/scripts/run_fig1_uniform.sh
#
# Copy back only synthetic.json and config.json -- checkpoints are 34 MB each
# and nothing downstream reads them.
set -euo pipefail
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
[ -d "$CACHE" ] || { echo "no cohort cache at $CACHE" >&2; exit 1; }

# b128.yaml sets profile: full and batch_size: 128.
# --rare-upweight on EVERY arm, and no --no-early-stop anywhere so early
# stopping stays on for all six. That uniformity is the point of this script.
# --no-eval skips the end-of-run privacy suite, which is the slow part and
# which nothing here reads; Test 1 scores from synthetic.json below.
BASE="--config examples/fedpyhealth/sweeps/b128.yaml --cohort-cache $CACHE
      --local-epochs 2 --n-rounds 50 --rare-upweight --no-eval"

REPS="${REPS:-1 2 3}"
ARMS="${ARMS:-local centralized fedavg ft_full ft_lora fedlibra}"
TAG=rwes128

for rep in $REPS; do
  for arm in $ARMS; do
    case "$arm" in
      local)       ARGS="--regime local" ;;
      centralized) ARGS="--regime centralized" ;;
      fedavg)      ARGS="--regime fedavg" ;;
      ft_full)     ARGS="--regime fedavg_ft --ft-epochs 20
                         --adapter none --adapter-rank 8" ;;
      ft_lora)     ARGS="--regime fedavg_ft --ft-epochs 20
                         --adapter lora_head --adapter-rank 8" ;;
      fedlibra)    ARGS="--regime fedavg_ft --ft-epochs 20
                         --irm-rho 100 --irm-warmup 17 --latent-dim 16 --xm-k 4
                         --adapter lora_head --adapter-rank 8" ;;
      *) echo "unknown arm $arm" >&2; exit 1 ;;
    esac

    NAME="${TAG}rep${rep}_${arm}"
    DIR="_outputs/${NAME}_save"
    if [ -f "$DIR/synthetic.json" ]; then
      echo "### skip $NAME -- synthetic.json already there"
      continue
    fi
    echo
    echo "############ $NAME  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/train.py $BASE $ARGS --run-name "$NAME" --resume
    echo "### generating $NAME at 8000/hospital  ($(date +%H:%M:%S))"
    python examples/fedpyhealth/generate.py --save-dir "$DIR" \
        --cohort-cache "$CACHE" --synth-per-hospital 8000
  done
done

# ---- score every finished run in one pass (CPU only) ----
RUNS=""
for rep in $REPS; do
  for arm in $ARMS; do
    d="_outputs/${TAG}rep${rep}_${arm}_save"
    [ -f "$d/synthetic.json" ] && RUNS="$RUNS --run ${arm}_rep${rep}=$d"
  done
done
[ -n "$RUNS" ] || { echo "nothing to score" >&2; exit 1; }

mkdir -p _outputs/results/tests
# 200 bootstraps, not the default 5 -- at 5 the spread is pure quantisation
# noise and swamps every difference worth reporting.
echo
echo "############ Test 1 SPECIALIST, 200 bootstraps  ($(date +%H:%M:%S))"
CUDA_VISIBLE_DEVICES= python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out "_outputs/results/tests/test1_${TAG}_spec_b200.json"

echo "############ Test 1 GENERALIST, 200 bootstraps  ($(date +%H:%M:%S))"
CUDA_VISIBLE_DEVICES= python examples/fedpyhealth/test1_prevalence.py \
    --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --n-bootstraps 200 --bootstrap-seed 0 \
    --out "_outputs/results/tests/test1_${TAG}_gen_b200.json"

echo
echo "all done ($(date +%H:%M:%S))"
echo
echo "Plot with a variant of examples/fedpyhealth/scripts/fig_frontier_rw.py."
echo "Aggregate as the main table does: mean over the 8 hospitals of"
echo "PrevVal_All_Prevalence_R2, then mean and spread over the three runs."
echo "Median, or the rare-code columns, give different numbers -- the four"
echo "statistics disagree by more than most arm-to-arm gaps."
