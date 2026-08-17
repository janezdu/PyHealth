#!/bin/bash
# Test 2, all four regimes, uncapped budget. The headline experiment.
#
#   sbatch examples/fedpyhealth/scripts/test2_4arm.sbatch
#
# RESUMABLE. Every finished classifier is written to
# <out>.partial immediately, and --resume replays them. If this hits the wall
# clock, just submit the exact same command again -- it picks up where it
# stopped. Do not raise --time speculatively; resubmitting is free.
#
# 27 arms x 4 mask folds = 108 classifiers:
#   real_pooled            1     all 8 hospitals' real train splits (TRTR ceiling)
#   real_local:<hid>       8     each hospital's own real data (the bar to beat)
#   tstr:centralized       1     one pooled generator, 16000 patients
#   tstr:fedavg            1     one federated generator, 16000 patients
#   tstr:fedavg_ft:<hid>   8     8 fine-tuned generators, 2000 each
#   tstr:local:<hid>       8     8 isolated generators, 2000 each
#
# --train-budget 0 is UNCAPPED: every arm trains on all the data it has, so
# hospital 438 pits its 79 real records against 2000 synthetic ones. A tstr win
# is therefore a win on DATA AVAILABLE, not on per-record quality -- which is
# the intended question. The records column in the output table reports what
# each arm actually got.
#
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-test2-4arm
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail

source .venv/bin/activate

# Pre-flight. Scoring a run whose synthetic.json predates the uniform-count
# change would silently mix a 68-patient arm in with 2000-patient ones, and the
# table would look fine. Cheaper to refuse here than to discover it afterwards.
python - <<'PY'
import json, os, sys
sys.path.insert(0, "examples/fedpyhealth")
from utils.cohort import load_shared_generator

expect = {"centralized": (True, 16000), "fedavg": (True, 16000),
          "fedavg_ft": (False, 2000), "local": (False, 2000)}
bad = []
for regime, (want_shared, want_n) in expect.items():
    d = f"_outputs/{regime}_full_strat8_random_save"
    p = f"{d}/synthetic.json"
    if not os.path.exists(p):
        bad.append(f"{regime}: no synthetic.json"); continue
    blob = json.load(open(p))
    sizes = sorted({len(v) for v in blob["per_hospital"].values()})
    shared = load_shared_generator(d)
    if shared != want_shared:
        bad.append(f"{regime}: shared_generator={shared}, expected {want_shared}")
    if sizes != [want_n]:
        bad.append(f"{regime}: per-hospital sizes {sizes}, expected [{want_n}]")
    print(f"  {regime:12s} shared={shared!s:5s} per-hosp={sizes}")
if bad:
    print("\nPRE-FLIGHT FAILED:", *bad, sep="\n  ")
    sys.exit(1)
print("pre-flight OK")
PY

python examples/fedpyhealth/test2_rare_efficacy.py \
    --run centralized=_outputs/centralized_full_strat8_random_save \
    --run fedavg=_outputs/fedavg_full_strat8_random_save \
    --run fedavg_ft=_outputs/fedavg_ft_full_strat8_random_save \
    --run local=_outputs/local_full_strat8_random_save \
    --mask-folds 4 \
    --train-budget 0 \
    --resume \
    --out _outputs/results/tests/test2_4arm_uncapped.json \
    "$@"
