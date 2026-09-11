# PR draft note — caching for the EHR-generation examples

Running record of what we've established, for the PR description. Lives on
`rare-code-fed-eval` so it never lands in the PR itself; the PR is built in the
worktree at `/projects/bgyw/janezdu/PyHealth-cache` (`ehrgen-cache` @ 89bb9ea,
upstream master).

**Status: premise DISPROVED (2026-09-08). Upstream already solves this.**
No caching PR. See the results and verdict below.

---

## The claim to be tested

Running an EHR-generation example twice re-does a per-patient extraction loop
over the whole dataset that could be read from disk instead.

## What upstream already does (found before writing any code)

`set_task` on upstream master already has a content-keyed cache
(`base_dataset.py`, added in `2ed2ba4`):

```
{cache_dir}/tasks/{task_name}_{uuid5(task_params)}/
    task_df.ld/                        # the per-patient extraction loop
    samples_{uuid5(proc_params)}.ld/   # processed samples + schema.pkl
```

- keyed on task params **and** processor params
- `FileLock`-guarded so parallel jobs don't race to build it
- validated by `index.json`, which litdata writes only after all chunks flush
- default `cache_dir` is a persistent `platformdirs` path (`~/.cache/pyhealth/`)

So a second launch *should* already skip the loop. This is the thing to
disprove before proposing a cache.

## Measurement

`scratchpad/bench_settask.py` — times dataset load and `set_task` separately and
installs a log handler that catches `Found cached ... / skipping task
transformation`. Run twice against one fresh `--cache-dir`.

Gotchas already hit (keep for the PR's repro section):

- The venv installs pyhealth **editable via a plain `.pth`** → a script outside
  the worktree loads the *main* clone. Needs `PYTHONPATH=<worktree>`; `cd`
  alone is not enough (`sys.path[0]` is the script's dir, not cwd). The script
  prints `pyhealth from:` as its first line for this reason.
- Local MIMIC-III is an **incomplete download** — everything alphabetically
  after `INPUTEVENTS_CV.csv.gz` is missing, including `PATIENTS.csv.gz`.
  Benchmarking on MIMIC-IV 3.1 instead (complete, and larger: 33 MB
  `diagnoses_icd.csv.gz` vs 4.5 MB).
- **Any script calling `set_task` needs an `if __name__ == "__main__":`
  guard.** dask and litdata spawn workers by re-importing the module, so
  top-level work re-runs in every child -- a fork bomb, not a slow run.
  pyhealth warns about it (`set_task method accessed from a non-main
  process`), which is easy to miss in the log spam. `halo_mimic3.py` already
  has the guard; anything we add to the docs should keep it.
- `num_workers` defaults to **1** (`base_dataset.py:336`) and the event
  transform runs `n_workers=self.num_workers, threads_per_worker=1`
  (`base_dataset.py:574-577`), so the default path uses one core no matter what
  the allocation has. That default is also the honest baseline, since the
  examples don't pass it.
- `MIMIC4Dataset` takes `ehr_root`/`ehr_tables`; `MIMIC3Dataset` takes
  `root`/`tables`.
- **Both MIMIC datasets force three tables regardless of what you pass**:
  `default_tables = ["patients", "admissions", "icustays"]` (`mimic3.py:56`,
  `mimic4.py:70`). So a "diagnoses only" run still needs all three on disk.
  Local MIMIC-IV `icu/` is empty (no `icustays.csv.gz`), and local MIMIC-III is
  missing only `PATIENTS.csv.gz` -- hence MIMIC-III, after fetching that one
  2.6 MB file.

### Results (MIMIC-III 1.4, `dev=True`, diagnoses only)

| phase | cold | warm |
|---|---|---|
| dataset load | _TBD_ | _TBD_ |
| `set_task` | _TBD_ | _TBD_ |
| cache hits logged | _TBD_ | _TBD_ |

`dev=False` not measured yet — deliberately.

## Which PR this becomes, depending on the warm run

1. **warm hits, fast** — upstream already handles it. No caching PR. Our
   `utils/cohort.py` cache stays justified by its *other* properties (torch-free
   parquet reads for Test 1/Test 2, per-hospital fold layout as the split, a
   pinned processor shared across FedAvg clients) — none of which are
   upstream-able as-is.
2. **warm misses because `~/.cache` is small/purged on Delta** — the fix is a
   `cache_dir=` argument in the examples pointing at scratch. ~5 lines, real,
   but a different PR than a new cache.
3. **warm misses because the cache key churns** — something in `task_params` or
   `proc_params` differs run to run. That's an upstream bug and the most
   valuable of the three.

## Design that is now moot (kept only as a record)

Settled before measuring; did NOT survive, since the cache it would add
already exists in `set_task`.

- Helpers go in `pyhealth/tasks/generate_ehr.py` beside `decode_dataset` — the
  cache is the on-disk form of this task's output, not an eICU trick, and that
  placement sits *above* the `root` vs `ehr_root` signature split so one
  implementation serves MIMIC-III, MIMIC-IV and eICU.
- `NestedSequenceProcessor` needs real `save()`/`load()`; the base class
  declares them as no-op hooks. Full state is `code_vocab`, `_next_index`,
  `_max_inner_len`, `_padding` — dropping `_max_inner_len` round-trips to
  differently-shaped tensors, since it sets the padded width of every visit.
- Layout: `samples.parquet` (patient_id, visit_idx, pos, code, + passthrough
  columns), `processor.json`, `manifest.json` with `task_name`.
- **Codes as strings, not vocabulary indices.** A vocabulary change then becomes
  a detectable mismatch instead of a silent decode against wrong indices.
- **Passthrough columns.** `EHRGenerationEICU` already emits `hospital_id`
  alongside `visits` upstream; a cache storing only `visits` would silently drop
  it. ~3 lines, testable with synthetic samples, not speculative.
- **`task_name` in the manifest** so a MIMIC cache can't be loaded as an eICU
  one and quietly mix ICD-9 vocabularies.
- Opt-in everywhere. Default off ⇒ the examples behave exactly as today.
- `halo_mimic3.py` has no argparse (flat top-to-bottom script) — a `CACHE_DIR`
  constant beside `root=`, not a parser, keeps it readable.

### Scope decisions already made

- **One PR**, not two.
- **fedpyhealth stays out.** MIMIC only.
- Target `upstream/master` — upstream has **no `develop` branch**, despite what
  `.llms/rules/02-contributing.md` says.
- Not `git cherry-pick`: the source commits (`1b53deb`, `e6ab604`) mix caching
  with adapters/IRM. Port by hand from a scratch copy of `utils/cohort.py`.
- Per `CONTRIBUTING.md`, needs tests + docs. Test can import the example
  directly — `tests/core/test_fedpyhealth_cohort.py` sets that precedent with a
  `sys.path.insert`.

## Framing for the PR description

The saving is per *launch*, so the number that lands is the multiplier, not the
ratio: "X min → Y s per launch; an N-job sweep goes from A to B." Report
`dev=True` (what a newcomer runs — shows the flag isn't overhead) and
`dev=False` (what justifies it). Note that `dev` truncates to 1000 patients
*after* the CSV parse, so the parse cost is paid either way — only the task loop
scales.
