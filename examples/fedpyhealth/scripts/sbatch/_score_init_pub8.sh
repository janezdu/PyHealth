#!/bin/bash
# Test 1 (prevalence fidelity) for the pub8 warm-start experiment, on hilo8 test.
#
# Zero-shot is the pub8 checkpoint's OWN synthetic set, scored against hilo8 with
# no training on hilo8. pub8 is one shared generator, so its synthetic.json
# holds one 16,000-patient set repeated under each pub8 hospital id. The scorer
# looks sets up by hilo8 id and SKIPS any hospital it cannot find -- scored
# as-is, all eight would be skipped. So the same set is re-keyed to the hilo8
# ids first. That is exact, not an approximation: it is what train.py writes
# for any shared generator.
#
# Scores whichever init_pub8_* arms have finished, so it can run before them
# (zero-shot only) and again after them. TAG names the output files.
#SBATCH --account=bgyw-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --job-name=score-init-pub8
#SBATCH --time=02:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
: "${FEDCOHORT_CACHE:?set FEDCOHORT_CACHE to the directory holding your cohort caches}"
: "${INIT_FROM:=_outputs/centralized_full_pub8_nrounds25_localepochs2_save}"
: "${TAG:=all}"
: "${ARMS:=centralized local fedavg fedavg_lora}"
CACHE="$FEDCOHORT_CACHE/hilo8_random"
ZS=_outputs/init_pub8_zeroshot_save
RES=_outputs/results/tests
mkdir -p "$RES"

SRC="$INIT_FROM" ZS="$ZS" CACHE="$CACHE" python -I - <<'EOF'
import json, os
src, dst, cache = os.environ["SRC"], os.environ["ZS"], os.environ["CACHE"]
blob = json.load(open(os.path.join(src, "synthetic.json")))
sets = list(blob["per_hospital"].values())
assert blob["shared_generator"] and all(s == sets[0] for s in sets), \
    "source is not one shared generator; re-keying would be wrong"
hids = list(json.load(open(os.path.join(cache, "manifest.json")))["hospitals"])
blob["per_hospital"] = {h: sets[0] for h in hids}
os.makedirs(dst, exist_ok=True)
json.dump(blob, open(os.path.join(dst, "synthetic.json"), "w"))
print(f"zero-shot: {len(sets[0])} synthetic patients re-keyed to {hids}")
EOF

RUNS="--run zeroshot=$ZS"
for arm in $ARMS; do
    dir=_outputs/init_pub8_${arm}_save
    if [ -f "$dir/synthetic.json" ]; then RUNS="$RUNS --run $arm=$dir"
    else echo "SKIP $arm -- no synthetic.json at $dir" >&2; fi
done
# Scratch reference: same budget, same weighting, no warm start.
RUNS="$RUNS --run scratch_fedavg=_outputs/dp_ctrl_fedavg_save"
# Arms outside the init_pub8_* naming, as "name=dir" pairs.
for pair in ${EXTRA:-}; do
    if [ -f "${pair#*=}/synthetic.json" ]; then RUNS="$RUNS --run $pair"
    else echo "SKIP $pair -- no synthetic.json" >&2; fi
done
echo "scoring:$RUNS"

# synth-cap 2000: local emits 2000 per site (full profile), so every arm is
# capped to that to stay matched.
for scope in hospital pooled; do
    [ "$scope" = hospital ] && tag=spec || tag=gen
    python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
        --fold test --synth-cap 2000 --rare-scope pooled --real-scope "$scope" \
        --n-bootstraps 200 --bootstrap-seed 0 \
        --out "$RES/test1_init_pub8_${TAG}_${tag}_b200.json"
done
echo done
