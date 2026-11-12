#!/bin/bash
# Every Figure 1 arm, three independent runs each, on ONE machine at ONE batch
# size. No SLURM -- this is for a workstation with a GPU and no queue.
#
# WHY ONE MACHINE. The published figure mixes batch sizes across arms (five at
# 128, FedLIBRA at 256) and mixes code versions (Aug vs Sep). That is why two
# retrained runs of an arm land on each other and nowhere near the published
# point. These 18 runs share a batch size, a code version, a cohort cache and
# a GPU, so their spread is run-to-run variation and nothing else.
#
# Batch 128, because sweeps/b128.yaml is the project's standing choice:
# "every result in this project so far was produced at 128."
#
# PER-ARM SETTINGS COPIED FROM EACH PUBLISHED RUN'S config.json. centralized
# is the odd one -- the only arm published WITH rare-code upweighting and WITH
# early stopping, so it alone does not take --no-early-stop.
#
# --metrics is eval-only and does not touch the trained weights, so it is left
# off here: it costs minutes per run and nothing downstream reads it. Test 1
# and Test 2 both score from synthetic.json afterwards.
#
# ~15 min a run on a single modern GPU, so ~4.5 h for all 18. Safe to leave
# overnight: --resume means an interrupted run continues rather than restarts,
# and finished runs are skipped.
#
#   export FEDCOHORT_CACHE=$HOME/cache/fedcohort      # holds hilo8_random/
#   bash examples/fedpyhealth/scripts/run_fig1_reps_local.sh
#
# Then copy ONLY synthetic.json and config.json back -- the checkpoints are
# 34 MB each and nothing downstream needs them:
#
#   rsync -av --include='*/' --include='synthetic.json' --include='config.json' \
#         --exclude='*' _outputs/ \
#         janezdu@dt-login03.delta.ncsa.illinois.edu:/projects/bgyw/janezdu/PyHealth/_outputs/
set -euo pipefail
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
[ -d "$CACHE" ] || { echo "no cohort cache at $CACHE" >&2; exit 1; }

BASE="--config examples/fedpyhealth/sweeps/b128.yaml --cohort-cache $CACHE
      --local-epochs 2 --n-rounds 50"

REPS="${REPS:-1 2 3}"
ARMS="${ARMS:-local centralized fedavg ft_full ft_lora fedlibra}"

for rep in $REPS; do
  for arm in $ARMS; do
    case "$arm" in
      local)       ARGS="--regime local --no-early-stop" ;;
      # the ONLY arm published upweighted and early-stopped
      centralized) ARGS="--regime centralized --rare-upweight" ;;
      fedavg)      ARGS="--regime fedavg --no-early-stop" ;;
      ft_full)     ARGS="--regime fedavg_ft --no-early-stop --ft-epochs 20
                         --adapter none --adapter-rank 8" ;;
      ft_lora)     ARGS="--regime fedavg_ft --no-early-stop --ft-epochs 20
                         --adapter lora_head --adapter-rank 8" ;;
      fedlibra)    ARGS="--regime fedavg_ft --no-early-stop --ft-epochs 20
                         --irm-rho 100 --irm-warmup 17 --latent-dim 16 --xm-k 4
                         --adapter lora_head --adapter-rank 8" ;;
      *) echo "unknown arm $arm" >&2; exit 1 ;;
    esac

    NAME="pcrep${rep}_${arm}"
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

echo
echo "all done ($(date +%H:%M:%S))"
echo "runs produced:"
ls -d _outputs/pcrep*_save 2>/dev/null | sed 's|^|  |'
