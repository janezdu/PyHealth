# fedpyhealth workflow

You are an expert ML engineer whose job is to assist running experiments.

Use this skill when working on the fedpyhealth eICU pipeline in this repository. Your job is to help the user choose the correct command for the current stage of the experiment, keep the workflow reproducible, and write correct SLURM job scripts for this cluster.

## Core rule

**The cohort cache is the dataset.** One directory, built once by `utils/cohort.py`, holds the cohort's patients as `<hospital>.<fold>.parquet` — 8 hospitals x train/val/test = 24 files — plus the pinned code vocabulary and a manifest. Every training and scoring job reads it and nothing else touches eICU.

That layout *is* the split: `420.train.parquet` is hospital 420's train fold, so there is no separate manifest of patient ids that can drift out of sync with the data. Rebuilding the cache re-splits the data, which invalidates checkpoints and makes finished runs incomparable — so build it once and keep pointing every job at the same directory.

Scoring runs on the **val** fold during development. Test is not read by any script yet; it stays clean for the final numbers.

## The files

| file | what it does |
|---|---|
| [main.py](examples/fedpyhealth/main.py) | the launcher — `train`, `test1`, `test2`, `all`, `status`. Generates and submits the SLURM job itself |
| [utils/cohort.py](examples/fedpyhealth/utils/cohort.py) | builds the cohort cache (~1-2h, once); also the read API every other script imports |
| [train.py](examples/fedpyhealth/train.py) | trains one regime; scores Test 1 in-run |
| [test1_prevalence.py](examples/fedpyhealth/test1_prevalence.py) | prevalence fidelity, standalone re-scoring |
| [test2_rare_efficacy.py](examples/fedpyhealth/test2_rare_efficacy.py) | rare-code ML efficacy (TSTR) |
| [exp_log.py](examples/fedpyhealth/exp_log.py), [results.py](examples/fedpyhealth/results.py) | run registry, leaderboard |
| `scripts/run_*.sh` | standalone sbatch wrappers, for when the user wants to submit without `main.py` |

## Command selection

1. **Build the cache.** `utils/cohort.py` — one eICU pass that selects the hospitals, computes rare codes, splits 70/10/20, writes the Parquet files, and verifies they reproduce the eICU tensors. Run once. Use when the user says "make a cohort", "pick hospitals", or "choose 2 from each size band".
2. **Train.** `main.py train --regime <regime> --profile tiny|full`, or `main.py all` for the four-regime table.
3. **Score.** `main.py test1` and `main.py test2`, after training, since both read each run's saved `synthetic.json`.
4. **Check on jobs.** `main.py status` — parses `squeue` plus the log tails.

## Default workflow

```bash
# 0. paths come from the environment, never hardcoded (data-safety rule)
export EICU_ROOT=/path/to/eicu-crd/2.0
export FEDCOHORT_CACHE=/fast/scratch/fedcohort   # optional; defaults under _outputs/

# 1. build both cohort caches once (~1-2h, one job)
sbatch examples/fedpyhealth/scripts/run_cohort.sh

# builds strat8 (rare-code stratified) and strat8_random (plain shuffle) over
# the same 8 hospitals, then prints the coverage report for each

# 2. train against the cache
python examples/fedpyhealth/main.py train --profile tiny --dry-run   # inspect first
python examples/fedpyhealth/main.py all --profile full

# 3. score the saved runs (CPU-cheap: reads the cache, not eICU)
python examples/fedpyhealth/main.py test1 --run fedavg=_outputs/<run_name>_save
python examples/fedpyhealth/main.py test2 --run fedavg=_outputs/<run_name>_save
```

Use `$FEDCOHORT_CACHE/strat8` unless the user explicitly asks for another. Keep the same cache directory across training and evaluation so sample assignment stays identical across regimes.

## Writing a SLURM job script for this cluster

`main.py` generates its own job scripts, so prefer it. Write one by hand only for a job `main.py` does not cover — building the cache is the main one. Copy this preamble; each line is load-bearing on NCSA Delta.

```bash
#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgyw-delta-gpu
#SBATCH --job-name=fed-<what-it-does>
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=_outputs/slurm/%x-%j.out

set -euo pipefail

source .venv/bin/activate

python examples/fedpyhealth/<script>.py "$@"
```

Rules that are easy to get wrong:

- **Always request a GPU, even for CPU-only work.** Delta rejects any zero-GPU job submitted under a `*-delta-gpu` account, on the `cpu` partition and GPU partitions alike, and this project only holds GPU accounts. A CPU-only job (building the cache, EDA) should request one idle GPU on `gpuA40x4` — the cheapest tier and usually the shortest queue. Training uses `gpuA100x4`.
- **`mkdir -p _outputs/slurm` before submitting.** `--output` does not create its directory; the job dies at submit time if it is missing.
- **`source .venv/bin/activate`, not pixi.** The repo documents pixi, but this clone runs from `.venv`.
- **Submit from the repo root.** Every path in these scripts is repo-relative, and `main.py` refuses to run from anywhere else.
- **`"$@"` at the end** so per-knob overrides pass through: `sbatch scripts/run_train_full.sh --n-rounds 100`.
- **Size the wall clock to the work.** Building the cache ~2h. A full FedAvg run trains clients sequentially on one GPU, so budget 48h. Test 2 trains a classifier per hospital per mask fold — 6h is generous.
- **Point `FEDCOHORT_CACHE` at fast local storage** (on Delta, `/work/nvme`), not the repo and not `/work/hdd`. It is read at the start of every job.
- **Everything else writes under `_outputs/`**, which is gitignored: SLURM logs, checkpoints, `synthetic.json`, results.

For resumable training, checkpoint every round and resubmit with `--resume`; chain windows with `sbatch --dependency=afterany:<jobid>`. Chain scoring after training with `--dependency=afterok:<id1>:<id2>...`, which `main.py all` does automatically.

## Key reminder

Building the cache and running experiments are different jobs with different lifetimes. The workflow is only reproducible when the cache is built once and reused for every later training and scoring job.
