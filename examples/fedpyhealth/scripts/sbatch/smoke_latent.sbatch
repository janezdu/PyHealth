#!/bin/bash
# Validate the per-patient latent before spending on it.
#
# WHAT THE LATENT IS. z ~ N(0, I), one draw per PATIENT, projected into the
# embedding at position 1 -- the conditioning slot HALO reserves for a label and
# this port leaves empty. Position 1 is not arbitrary: the fine head pairs
# history[t] with visits[t+1], so position 1's hidden state is exactly what
# predicts visit 1, and 79% of this cohort's patients have only visit 1.
#
# NO ENCODER, NO KL. z is always drawn from the prior, so training and
# generation see the same distribution by construction and there is no
# posterior gap. That also means an ordinary loss would simply AVERAGE over z
# and ignore it -- best-of-K is what rewards the model for making z mean
# something, because if z spreads the predictions coherently then one of the K
# candidates lands close and that is the branch that gets the gradient.
#
# WHAT TO READ. Dropout is 0 here on purpose, so z is the ONLY source of
# variation and xm_spread measures exactly one thing: whether the model learned
# to use the latent. With dropout it measured noise sensitivity instead, which
# is why that probe could not answer this question.
#
#   spread -> 0        z is being ignored; the latent is dead weight and
#                      best-of-K is training on 1/K of the gradient.
#   spread grows       z is doing work.
#
# Step 3 checks the thing that would silently corrupt everything: a checkpoint
# trained WITH a latent must not load into a model built without one. z_proj has
# parameters, unlike dropout, so generate.py has to rebuild it from config.json.
#
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=smoke-latent
#SBATCH --time=00:40:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate
CACHE="$FEDCOHORT_CACHE/hilo8_random"

echo "############ 1/3  latent OFF must be bit-unchanged"
python examples/fedpyhealth/train.py --profile tiny --regime centralized \
    --cohort-cache "$CACHE" --run-name smoke_z_off --no-tb 2>&1 \
    | grep -E "xm K=|Saved results|Traceback|Error" | head -5
echo "  (no 'xm K=' lines = correctly off)"

echo
echo "############ 2/3  latent 16, K=4, dropout 0 -- z is the only variation"
python examples/fedpyhealth/train.py --profile tiny --regime centralized \
    --cohort-cache "$CACHE" --latent-dim 16 --xm-k 4 \
    --run-name smoke_z16 --no-tb 2>&1 \
    | grep -E "xm K=|Saved results|Traceback|Error" | head -20

echo
echo "############ 3/3  the checkpoint round-trip"
python examples/fedpyhealth/generate.py \
    --save-dir _outputs/smoke_z16_save --cohort-cache "$CACHE" \
    --synth-per-hospital 32 2>&1 | tail -4
echo "  (if that loaded, generate.py rebuilt z_proj from config.json)"
