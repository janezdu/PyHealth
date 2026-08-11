# fedpyhealth workflow

You are an expert ML engineer whose job is to assist running experiments.

Use this skill when working on the fedpyhealth eICU pipeline in this repository. Your job is to help the user choose the correct command for the current stage of the experiment, keep the workflow reproducible, and write correct SLURM job scripts for this cluster.

## Core rule

Do not treat the cohort selection file as the final dataset freeze.

- `*.cohort.json` selects hospitals only.
- the frozen `strat8.json`-style manifest contains the actual patient-level train/val split and is what training and scoring consume.

If the user wants the same samples every time they rerun experiments, the frozen manifest must be reused. `utils/cohort_io.load_manifest` enforces this: it rejects a file that has no `train_patient_ids`/`val_patient_ids` rather than letting a selection file through.

## The files

| file | what it does |
|---|---|
| [main.py](examples/fedpyhealth/main.py) | the launcher — `train`, `test1`, `test2`, `all`. Generates and submits the SLURM job itself |
| [prepare_dataset.py](examples/fedpyhealth/prepare_dataset.py) | `preview` (pick hospitals, seconds) and `freeze` (write the train/val manifest, ~1-2h) |
| [train.py](examples/fedpyhealth/train.py) | trains one regime; scores Test 1 in-run |
| [test1_prevalence.py](examples/fedpyhealth/test1_prevalence.py) | prevalence fidelity, standalone re-scoring |
| [test2_rare_efficacy.py](examples/fedpyhealth/test2_rare_efficacy.py) | rare-code ML efficacy (TSTR) |
| [exp_log.py](examples/fedpyhealth/exp_log.py), [results.py](examples/fedpyhealth/results.py) | run registry, leaderboard |
| `scripts/run_*.sh` | standalone sbatch wrappers, for when the user wants to submit without `main.py` |

## Command selection

1. **Pick hospitals.** `prepare_dataset.py preview` — fast, reads `patient.csv` only, no GPU, no PyHealth import. Use when the user says "find a cohort", "pick hospitals", or "choose 2 from each size band". `--list-only` prints candidates per band for hand-picking; otherwise it draws and writes the selection file.
2. **Freeze the split.** `prepare_dataset.py freeze` — reads the selection file and writes the manifest every regime reuses. This is the step that makes the dataset actually frozen. Run once.
3. **Train.** `main.py train --regime <regime> --profile smoke|full`, or `main.py all` for the four-regime table.
4. **Score.** `main.py test1` and `main.py test2`, after training, since both read each run's saved `synthetic.json`.

## Default workflow

```bash
# 1. choose the hospitals (seconds; writes cohorts/strat8.cohort.json)
python examples/fedpyhealth/prepare_dataset.py preview \
  --size-bands 0-199,200-499,500-1999,2000- --per-band 2 --seed 2

# 2. freeze the exact train/val split once (writes cohorts/strat8.json)
python examples/fedpyhealth/prepare_dataset.py freeze

# 3. train on the frozen manifest
python examples/fedpyhealth/main.py train --profile smoke --dry-run   # inspect first
python examples/fedpyhealth/main.py train --profile full

# 4. score the saved runs
python examples/fedpyhealth/main.py test1 --run fedavg=_outputs/<run_name>_save
python examples/fedpyhealth/main.py test2 --run fedavg=_outputs/<run_name>_save
```

Use the stratified 8 manifest by default; override only when the user explicitly asks. Keep the same frozen manifest path on training and evaluation so sample assignment stays identical across regimes.

## Writing a SLURM job script for this cluster

`main.py` generates its own job scripts, so prefer it. Write one by hand only for a job `main.py` does not cover. Copy this preamble — each line is load-bearing on NCSA Delta.

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

- **Always request a GPU, even for CPU-only work.** Delta rejects any zero-GPU job submitted under a `*-delta-gpu` account, on the `cpu` partition and GPU partitions alike, and this project only holds GPU accounts. A CPU-only job (the freeze step, EDA) should request one idle GPU on `gpuA40x4` — the cheapest tier and usually the shortest queue. Training uses `gpuA100x4`.
- **`mkdir -p _outputs/slurm` before submitting.** `--output` does not create its directory; the job dies at submit time if it is missing.
- **`source .venv/bin/activate`, not pixi.** The repo documents pixi, but this clone runs from `.venv`.
- **Submit from the repo root.** Every path in these scripts is repo-relative, and `main.py` refuses to run from anywhere else.
- **`"$@"` at the end** so per-knob overrides pass through: `sbatch scripts/run_train_full.sh --n-rounds 100`.
- **Size the wall clock to the work.** Smoke ~1h; a full FedAvg run trains clients sequentially on one GPU, so budget 48h. Test 2 trains a classifier per hospital per mask fold — 6h is generous.
- **Everything writes under `_outputs/`**, which is gitignored: SLURM logs, checkpoints, `synthetic.json`, results.

For resumable training, checkpoint every round and resubmit with `--resume`; chain windows with `sbatch --dependency=afterany:<jobid>`. Chain scoring after training with `--dependency=afterok:<id1>:<id2>...`, which `main.py all` does automatically.

## Key reminder

The hospital list and the frozen dataset are different artifacts. The workflow is only reproducible when the frozen manifest is created once and reused for every later training and scoring job.
