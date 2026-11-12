#!/bin/bash
# The MISSING latent arm: fedavg + structured latent z=16 + best-of-K K=4, NO IRM.
#
# WHY IT EXISTS. The frontier figure carries two best-of-K arms. Swapping them
# for latent equivalents needs both to exist, and only one does:
#
#   FedAvg + IRM + best-of-K  ->  lat_fed_irm   ALREADY TRAINED
#       _outputs/fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_z16_xm4_save
#   FedAvg + best-of-K        ->  THIS RUN      did not exist
#
# The only latent best-of-K without IRM on disk is lat_cent, which is
# CENTRALIZED, not federated, and is also broken (specialist R^2 -23.0). Using
# it would swap the regime as well as the mechanism, so it is not a substitute.
#
# DROPOUT STAYS 0. With xm_k=4 and dropout 0 the K candidates differ only
# through the per-patient latent draw -- which is the entire point of a "latent
# equivalent". Adding dropout would make it the same arm as fed_xm4_do03 with a
# latent bolted on, and the comparison would answer nothing.
#
# Every other knob is copied from the IRM sibling above so the pair differs in
# ONE thing. Diffed against its config.json: E2, R50, privacy metrics,
# --no-early-stop, z16, xm4, dropout 0, batch 256, lr 1e-4, 4 layers/4 heads.
#
# NOT LAUNCHED THROUGH main.py, deliberately. main.py builds its own --run-name
# (fedavg_full_hilo8_random_nes_localepochs2_nrounds50_...), which does not match
# the E2_R50 convention the other latent arms use, and it cannot be overridden
# without passing --run-name twice. Omitting --run-name here lets train.py's own
# make_run_name produce the conventional
#     fedavg_E2_R50_hilo8_random_privacy_nes_z16_xm4
# Resources are copied from main.py so this is the same job it would have built.
#
# WALL CLOCK, measured not guessed. The exact sibling (lat-fed-irm, job
# 21858470: z16 + IRM + xm4, E2/R50, same cohort) ran in 16m31s. Neighbours:
# xm01-fed-k4 14m50s, fed-xm4-do03 29m24s, slowest recent baseline 2h16m.
# 02:00:00 is ~4x the sibling, which covers K=4's extra forward passes with
# room to spare.
#
# Do NOT copy main.py's 48h default here. Its allowance is sized for the old
# pre-September full-cohort runs; on this cohort a 48h reservation only costs
# queue priority. (The 12-19h fed-fedavg-full jobs in sacct predate September
# and two of the three FAILED, so that elapsed is time-to-failure, not runtime.)
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --job-name=fed-lat-xm4
#SBATCH --time=02:00:00
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out
set -euo pipefail
source .venv/bin/activate

python examples/fedpyhealth/train.py --profile full --regime fedavg \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \
    --local-epochs 2 --n-rounds 50 --metrics privacy --no-early-stop \
    --latent-dim 16 --xm-k 4 \
    --resume
echo done
