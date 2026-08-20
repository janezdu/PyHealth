# notes

Reference material for the federated synthetic-EHR experiment. Prose and
numbers, not code — everything here is written for a person picking the project
back up, including the person who wrote it.

| file | what it is |
|---|---|
| [CHEATSHEET.md](CHEATSHEET.md) | the numbers you keep needing: cohort sizes, what "rare" means, generator and classifier hyperparameters, baseline floors |
| [TODO.md](TODO.md) | outstanding work, each item carrying the evidence that motivated it |
| [eicu-hospitals.md](eicu-hospitals.md) | the eICU hospital size distribution — which sites exist and how big, for designing a cohort |

## Related, elsewhere

- `../cohorts/*.config.json` — the git record of **which hospitals** each cohort
  used. Documentation, not input: nothing reads these files. The build path is
  `scripts/run_cohort.sh` → `utils/cohort.py` → a cache directory holding
  `manifest.json`, and every downstream job reads that manifest. The configs
  exist because the cache lives outside the repo.
- `../viz/` — the interactive pages, each regenerable from its `*_template.html`
  plus the script that renders it (`../eda.py` for `cohort_eda.html` and
  `rare_code_tail.html`; see `../viz/README.md`).
- `../configs/eda.yaml` — the standing EDA settings. `python eda.py --list`
  prints every analysis with its config keys and defaults.

## Regenerating

`eicu-hospitals.md` comes from `utils/eicu_sizes.py`, which reads only
`patient.csv` and takes seconds:

```bash
export EICU_ROOT=/path/to/eicu-crd/2.0
python examples/fedpyhealth/utils/eicu_sizes.py            # console summary
```

`CHEATSHEET.md` and `TODO.md` are maintained by hand. The cheatsheet cites its
sources per section, so it can be re-derived from `manifest.json`,
`PROFILES["full"]` in `train.py`, and the test argument parsers.

## A standing caution

Numbers in these files are tied to a specific cohort — `strat8_random` unless
stated otherwise. Rebuilding the cohort re-splits the data and redefines the
rare-code set, which makes every recorded number here stale at once. Check the
cohort name before trusting a figure.
