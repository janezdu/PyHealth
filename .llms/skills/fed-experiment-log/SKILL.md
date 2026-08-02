---
name: fed-experiment-log
description: Track fedpyhealth SLURM runs in an experiment log and sweep finished ones. Use when asked to check on launched runs, see which jobs finished, sweep/aggregate the latest runs into a leaderboard, or record a just-launched job. Stateful — it remembers which runs were already swept, so each check only processes what's new. Pairs with the fed-ehr-run / fed-eicu-settings launch skills and the slurm-inspect skill.
---

# fed-experiment-log — track + sweep federated-EHR runs

A thin, **stateful** layer over the fedpyhealth runs so launching and checking stay
in sync. Every launched run is appended to an append-only log
(`_outputs/experiment_log.jsonl`); a later *check* polls each unprocessed run's
SLURM state, **sweeps** (aggregates the metrics of) the ones that newly finished,
and marks them processed — so the next check only looks at what's new.

Engine: [examples/fedpyhealth/exp_log.py](../../../examples/fedpyhealth/exp_log.py)
(`record` / `check` / `list`). All operations are login-node-safe (an `sacct`
query + file IO + a no-torch aggregate). Run everything from the **repo root** with
`.venv/bin/python` (this clone runs via `.venv`, not pixi).

## The model

```
launch (sbatch) ──record──▶ experiment_log.jsonl ──check──▶ sacct status
                                                              │
                              processed=true ◀── sweep ───────┤ COMPLETED + results.json
                              (won't re-sweep)   (leaderboard) │
                              stays processed=false ◀──────────┘ RUNNING / PENDING
```

Each log line: `{job_id, run_name, config, results_path, status, processed, ...}`.
`run_name` is the identity that links a job to its `_outputs/results/<run_name>.json`
and its `_outputs/slurm/*<job_id>.out` log.

## When this skill is invoked

### "check my runs" / "what finished" / "sweep the latest runs"
This is the headline path. Run:
```bash
.venv/bin/python examples/fedpyhealth/exp_log.py check --sort Prevalence_R2
```
It prints: what's **still in flight**, anything that **finished without clean
results** (hand those to `slurm-inspect`), and a **leaderboard of the newly-finished
runs** (macro-averaged across hospitals, ranked by `--sort`). It then marks those
runs processed, so a later check won't re-sweep them.

- `--sort <metric>` — rank by any macro-avg metric (e.g. `Prevalence_R2`,
  `Privacy_Score`, `Prevalence_RMSE`). Default `Prevalence_R2`, descending.
- `--asc` — ascending (use for lower-is-better metrics like RMSE).
- `--all` — re-sweep **every** recorded run that has results, not just the new ones
  (use when the user wants the full standings, not just the delta).

Report back the leaderboard + the plain-language read (which config wins on fidelity
vs privacy, what's still running). Trust `Prevalence_R2` / `Privacy_Score`; the MLE
utility column is often collapsed in this pipeline — flag it, don't rank on it.

### "record this run" (usually automatic — see sync below)
```bash
.venv/bin/python examples/fedpyhealth/exp_log.py record \
    --job-id <JOBID> --config <path/to/config.yaml>
```
`run_name` is read from the config (sweep configs always pin a unique one). For a
run launched without a config file, pass `--run-name <name>` instead. Recording the
same `job_id` twice is a no-op, so it's safe to re-run.

### "list my experiments"
```bash
.venv/bin/python examples/fedpyhealth/exp_log.py list
```
Shows every tracked run with its last-known status and whether it's been processed.

## How it syncs with the launch skills

The launchers feed this log so a later `check` knows what to look up:

- **Sweeps (`make_sweep.py`)**: the generated `submit_all.sh` already does
  `sbatch --parsable` and `exp_log.py record` for every job — launching a sweep
  auto-populates the log. Just `bash _outputs/sweeps/<name>/submit_all.sh`.
- **`fed-ehr-run` / `fed-eicu-settings`** (single sbatch): after submitting, capture
  the job id and record it, e.g.
  ```bash
  jid=$(sbatch --parsable --job-name=fedavg_E2_R39_standard_8_utility ... run_ehr_eicu_full.sh ...)
  .venv/bin/python examples/fedpyhealth/exp_log.py record \
      --job-id "$jid" --run-name fedavg_E2_R39_standard_8_utility
  ```
  (Those skills note this step; the `run_name` must match the one `ehr_eicu.py`
  derives / you pass, so the `results_path` lines up.)

## Handoffs

- **Failures / "why did it die":** the check flags non-COMPLETED finals; diagnose
  with the **slurm-inspect** skill (`sacct`/`seff` + the `.out` log).
- **Launching / sizing / resuming:** **fed-ehr-run** (standard cohort) or
  **fed-eicu-settings** (smoke/full).
- **Manual deep dive on logs:** `compare_runs.py` parses the per-hospital tables
  straight from `.out` logs (useful for runs that predate the log or whose
  `results.json` is missing).

## Notes
- The log lives under `_outputs/` (gitignored) — it's per-clone state, not committed.
- Sweeping reads `_outputs/results/<run_name>.json`, written by `ehr_eicu.py` at the
  end of each run; a run that COMPLETED but has no results.json is flagged, not swept.
- Never run training here. This skill only queries + aggregates.
