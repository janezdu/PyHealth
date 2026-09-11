#!/bin/bash
# Train the delta-sparse adapters, then score them the same way as every other
# adapter arm.
#
#   sbatch examples/fedpyhealth/scripts/sbatch/_run_sparse_adapters.sh
#
# DO NOT run this before smoke_sparse_adapters.sbatch comes back clean. The one
# line to check there is each hospital's reported density: if the last_mlp_l1
# run reads 1.0000 the proximal step never fired and the arm is plain last_mlp
# wearing another name, which is exactly the failure these variants exist to
# avoid and is invisible in the loss.
#
# THE FEDAVG TRUNK IS NOT RETRAINED. Each save_dir is seeded with the
# fedavg_state.pt from the ft_epochs=20 sweep (40 rounds), so run_fedavg sees
# 50 >= 40 and skips straight to fine-tuning -- the same trick the IRM+FT runs
# used, and what makes these comparable to lora_head20 and last_mlp20 rather
# than to a differently-trained model.
#
# THE SPARSITY LADDER. last_mlp is 525,568 params, so:
#     density 0.5  -> 263K, still 7x lora_head. If this wins it is because
#                     sparsity REGULARISES, not because it is cheap.
#     density 0.1  ->  53K, the same order as lora_head's 38K.
#     density 0.07 ->  37K, a genuine like-for-like against lora_head, which is
#                     what makes "sparse vs low-rank at equal budget" sayable.
# THE L1 LADDER, and why it starts at 1 rather than at something small. The
# smoke run at lambda=0.1 came back at 0.993-0.999 density -- effectively dense.
# The soft-threshold is lr*lambda = 1e-4 * 0.1 = 1e-5 per step, against a mean
# |delta| of 162.5/33,088 = 0.005, so it was subtracting a tenth of a typical
# step and the gradient simply outran it. The scale that matters is the
# PER-COORDINATE GRADIENT, not the weight: Adam's update is ~lr whatever the
# gradient size, so lambda has to reach ~1 to compete with it at all and ~10 to
# put coordinates on zero. Hence 1 / 10 / 100.
#
# The L1 arms get no target density -- lambda sets it indirectly, and where it
# lands is a result rather than a setting. Read it off the per-hospital
# "delta density" lines in this job's log.
#
# THE SGD ARM at lr=1.0, four orders above the model's 1e-4. The smoke ran it at
# 0.01 and its ||D||_1 came back at 0.08 against Adam's 118 -- 1,500x smaller, so
# the site had barely fine-tuned at all and IHT was projecting noise. 1e-4 is an
# ADAM learning rate; SGD needs its own or the arm underperforms for reasons
# that have nothing to do with sparsity.
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=sparse-adapters
#SBATCH --time=04:00:00
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

CACHE="$FEDCOHORT_CACHE/hilo8_random"
TRUNK=_outputs/fedavg_ft_full_hilo8_random_nes_ftepochs20_adapternone_save/fedavg_state.pt
[ -f "$TRUNK" ] || { echo "no trunk at $TRUNK" >&2; exit 1; }

# --resume is what makes the seeded trunk actually get used. Without it
# run_fedavg ignores an existing fedavg_state.pt and retrains all 40 rounds, so
# every arm fine-tunes a DIFFERENT federated global and the comparison between
# adapters is confounded with the trunk draw (FedAvg over 40 rounds x 8 clients
# is not bit-reproducible even at a fixed seed -- the existing ft20 arms have
# visibly different trunk checksums). Skipping the trunk is also ~2x faster.
COMMON="--profile full --regime fedavg_ft --cohort-cache $CACHE \
        --no-early-stop --ft-epochs 20 --resume"

# name -> extra flags. Kept as one list so the seeding, training and scoring
# loops below cannot drift out of step with each other.
declare -a ARMS=(
  "iht50:--adapter last_mlp_iht --adapter-sparsity 0.5"
  "iht10:--adapter last_mlp_iht --adapter-sparsity 0.1"
  "iht07:--adapter last_mlp_iht --adapter-sparsity 0.07"
  "l1_1:--adapter last_mlp_l1 --adapter-l1 1"
  "l1_10:--adapter last_mlp_l1 --adapter-l1 10"
  "l1_100:--adapter last_mlp_l1 --adapter-l1 100"
  "iht10_sgd:--adapter last_mlp_iht --adapter-sparsity 0.1 --adapter-optim sgd --adapter-lr 1.0"
)

RUNS=""
for entry in "${ARMS[@]}"; do
    name="${entry%%:*}"
    flags="${entry#*:}"
    # train.py derives its own save_dir from the resolved config, so ask it
    # rather than reconstructing the name here and hoping the two agree.
    # tail -1 so any warning build_config prints ahead of the path cannot end
    # up concatenated into the directory name. stderr is left alone so a real
    # config error is still visible in the job log.
    dir=$(python examples/fedpyhealth/train.py $COMMON $flags --print-save-dir \
          | tail -1)
    if [ -z "$dir" ] || [ "${dir#_outputs/}" = "$dir" ]; then
        echo "could not resolve a save_dir for $name (got '$dir')" >&2
        exit 1
    fi
    mkdir -p "$dir"
    # Seed the trunk so the FedAvg half is skipped rather than retrained. Paired
    # with --resume in COMMON; the copy alone does nothing.
    [ -f "$dir/fedavg_state.pt" ] || cp "$TRUNK" "$dir/fedavg_state.pt"
    if [ -e "$dir/ft_"*.pt ] 2>/dev/null; then
        echo "$dir already holds fine-tuned checkpoints; --resume would keep" >&2
        echo "them and skip training. Move the directory aside first." >&2
        exit 1
    fi
    echo "############ training $name -> $dir"
    python examples/fedpyhealth/train.py $COMMON $flags
    echo "############ regenerating $name at 8000/hospital"
    python examples/fedpyhealth/generate.py --save-dir "$dir" \
        --cohort-cache "$CACHE" --synth-per-hospital 8000
    RUNS="$RUNS --run ${name}=${dir}"
done

echo "############ Test 1, SPECIALIST"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope hospital \
    --out _outputs/results/tests/test1_sparse_spec.json
echo "############ Test 1, GENERALIST"
python examples/fedpyhealth/test1_prevalence.py --cohort-cache "$CACHE" $RUNS \
    --fold test --synth-cap 8000 --rare-scope pooled --real-scope pooled \
    --out _outputs/results/tests/test1_sparse_gen.json
echo "############ Test 2, matched budget"
python examples/fedpyhealth/test2_rare_efficacy.py --cohort-cache "$CACHE" $RUNS \
    --fold test --train-budget 8000 \
    --out _outputs/results/tests/test2_sparse_budget8000.json
