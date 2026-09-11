#!/bin/bash
# Train + generate ONE swept arm. Parameterised by $ARM (a label) and $ARGS
# (train.py flags), passed in with `sbatch --export`.
#
# Each arm appends its resolved save_dir to $MANIFEST rather than the launcher
# guessing the name. train.py owns the naming rule; every time this project has
# rebuilt that rule in a shell script, an arm has silently gone unscored.
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"
FULL="--profile full --regime fedavg --cohort-cache $CACHE --no-early-stop $ARGS"

DIR=$(python examples/fedpyhealth/train.py $FULL --print-save-dir)
echo "arm=$ARM  save_dir=$DIR"

python examples/fedpyhealth/train.py $FULL
python examples/fedpyhealth/generate.py --save-dir "$DIR" \
    --cohort-cache "$CACHE" --synth-per-hospital 8000

# One line per arm; the scorer reads this instead of reconstructing names.
flock "$MANIFEST" -c "echo '$ARM=$DIR' >> '$MANIFEST'"
echo "done $ARM"
