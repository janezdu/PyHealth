# The standard experiment recipe

Every cohort goes through the same four stages. Follow this and two cohorts'
numbers sit on one axis; deviate and they quietly don't.

This is the **procedure**. For the reference numbers of a particular cohort see
[CHEATSHEET.md](CHEATSHEET.md); for which hospitals a cohort used, see
`../cohorts/*.config.json`.

---

## Stage 0 — pick the hospitals

Exact post-task sizes, straight from the eICU CSVs. Local python, ~2 minutes,
no SLURM:

```bash
python examples/fedpyhealth/utils/hospital_sizes.py \
    --bands 100-1499 --per-band 8 --seed 1 \
    --verify $FEDCOHORT_CACHE/hilo8_random/manifest.json
```

Prints a `--hospitals a,b,c` line. `--verify` checks the CSV shortcut reproduces
a known cohort's counts exactly; run it once before trusting a fresh draw.

> **Two size scripts exist and they do not agree.** `hospital_sizes.py` is the
> exact post-task count (`uniquepid`, code filter, cross-hospital drop) and is
> the one to size a band from. `utils/eicu_sizes.py` is a seconds-fast survey
> that counts `patienthealthsystemstayid` from `patient.csv` alone — a different
> key and no task filters, so it runs high. Use it to see the shape of the
> distribution, never to pick a threshold.

**Floor of 100 patients.** A hospital's rare codes are those held by ≥2 patients
but ≤5% of them, so below 40 patients no code can be both and the build dies.
At 40 "rare" means exactly 2 patients, sitting on the 5% line. 100 is the first
size where the band (2–5 patients) has real width.

## Stage 1 — build the cohort cache

**Local python, never sbatch.** It reads CSVs and writes Parquet; no model, no
GPU. Delta rejects zero-GPU jobs under a `*-delta-gpu` account, so an sbatch
wrapper has to book an idle GPU to satisfy the scheduler.

```bash
python examples/fedpyhealth/utils/cohort.py \
    --name <cohort> --hospitals <ids from stage 0> \
    --split random --seed 1 --out "$FEDCOHORT_CACHE/<cohort>"
```

Pin with `--hospitals` rather than re-running `--bands`: `draw_bands` samples
uniformly within a band, so the same seed over a different size table gives a
different cohort. Record the ids in `../cohorts/<cohort>.config.json`.

The build self-verifies — it rebuilds from the Parquet it just wrote and asserts
byte-identical tensors. Then, as a **separate** invocation (`--report` *replaces*
the build, it does not follow it):

```bash
python examples/fedpyhealth/utils/cohort.py --report --out "$FEDCOHORT_CACHE/<cohort>"
```

## Stage 2 — train the four regimes

```bash
python examples/fedpyhealth/main.py all --profile full \
    --config examples/fedpyhealth/sweeps/b128.yaml \
    --cohort-cache "$FEDCOHORT_CACHE/<cohort>"
```

Submits `fedavg`, `fedavg_ft`, `centralized`, `local`, then chains test1/test2
behind them with `afterok`. Batch 128 for comparability with every existing
result. If a training job fails its dependent tests sit in
`DependencyNeverSatisfied` forever — `scancel` them.

**The scoring `main.py` chains is a smoke check, not a result.** It runs with
every control off. Stage 3 is what produces publishable numbers.

## Stage 3 — score it properly

```bash
COHORT=<cohort> sbatch examples/fedpyhealth/scripts/score_cohort.sh
```

Four steps: regenerate at 8,000/hospital, Test 1 against **both** real targets,
Test 2 at a matched budget. The controls, and why each exists:

| control | why |
|---|---|
| `--synth-cap 8000` | Prevalence resolves only to 1/N. `fedavg`/`centralized` have no per-site ceiling — train.py hands each hospital the whole pooled generation — so uncapped they hold 8× what `local`/`fedavg_ft` do and win on resolution, not fidelity. |
| `--train-budget 8000` | Same confound on the utility axis. Uncapped, lo8's `centralized` scored AP 0.0209 against `real_pooled`'s 0.0192 — synthetic beating real, which is 16,000 classifier records against 2,684. |
| `--rare-scope pooled` | One shared definition of rare (the union of codes rare at any site). Per-hospital scope gives each site a different tail — 296 codes at one hilo8 site, 103 at another — so those numbers compare to nothing. |
| both `--real-scope` | Specialist and generalist are different questions and can rank the arms differently. Reporting one alone presents a choice of target as a fact about the generators. |

**Test 2 covers both levels too, in one run.** Its pooled arms are the cohort
view; the per-arm `own_site` block is each site scored against its own val rows.
No second invocation needed — but a consumer that only reads the top-level
numbers is seeing the cohort view alone.

### Why 8,000 per hospital, on any cohort

It is held **fixed across cohorts on purpose**, not derived from a cohort's own
sizes. hilo8_random was regenerated and capped at 8,000; lo8_random uses 8,000
even though its largest site holds 1,301 patients. Same 1/8000 prevalence
resolution, same classifier volume, one axis. Scaling it per cohort saves a
little GPU time and costs every cross-cohort comparison.

8,000 is also the largest cap that wastes nothing: `local` and `fedavg_ft` hold
exactly 8,000 per hospital and use all of it, while `fedavg` and `centralized`
come down from 64,000.

The one arm no cap can reach is `real_local` — a hospital holds what it holds.
It stays at its true size, which is the honest baseline.

## Stage 4 — render

Write `../viz/<cohort>.yaml` naming the three stage-3 outputs, then:

```bash
python examples/fedpyhealth/eda.py fidelity --config examples/fedpyhealth/viz/<cohort>.yaml
```

`../viz/lo8.yaml` is the worked example. Every source is named explicitly and
`cohort_name` is asserted against each file, so pointing a panel at another
cohort's run fails loudly instead of relabelling the numbers.

---

## Output filenames

Two mistakes this project has already made, both now prevented in code:

**The test scripts' default `--out` is cohort-agnostic.** `test1_prevalence.json`
and `test2_rare_efficacy.json` are claimed by a run on *any* cohort. Scoring
lo8_random silently replaced the hilo8_random results the dashboards were built
from — same shape, no error. `main.py` now derives `--out` from the cohort cache
name; `score_cohort.sh` names every file for the cohort. Never rely on the bare
default.

**A variant condition needs a `SUFFIX`.** The rare-upweighted (`_rw`) runs
overwrote a baseline result the same way. `score_cohort.sh` takes `SUFFIX=_rw`
and threads it through the run directories and every output name.

## Standing caveats

- **Val folds are thin at small sites.** Under ~25 patients, per-hospital early
  stopping is not a real signal and falls through to `--es-fallback`. Check what
  actually happened in the training logs before describing early stopping in a
  writeup.
- **Band rare-code results by support.** A code with one positive in a
  24-patient test fold gives an AP that is close to a coin flip. hilo8's pooled
  rare run had 32.7% of scored codes with zero test positives, which *depressed*
  the metrics rather than inflating them.
- **AP's floor is prevalence**, so the `global_rare` column sits far below
  `overall` for every arm including `prior`. That is arithmetic. Compare arms
  within a column, never across columns.
- **Adam state is discarded constantly.** `train_model` builds a fresh optimizer
  per call — 400 of them over a 50-round FedAvg — and fine-tuning starts clean
  too. Standard FedAvg, but worth stating, and it bites hardest at small sites
  that take one optimizer step per epoch.
