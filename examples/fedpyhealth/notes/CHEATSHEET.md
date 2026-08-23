# Federated synthetic-EHR experiment — cheatsheet

Reference numbers for the **active cohorts** and the four training regimes. Read
off the frozen cohort manifests and the run configs; if you rebuild a cohort or
change a profile, regenerate this (see [Provenance](#8-provenance)).

For the **procedure** — how to go from a size band to a rendered dashboard — see
[RECIPE.md](RECIPE.md). This file is the numbers; that file is the steps.

> **`strat8_random` is retired.** Earlier versions of this cheatsheet documented
> it (8,282 patients, 641 pooled rare codes, batch 64, `--fold val`). It was
> replaced by `hilo8_random` on 2026-08-19 because two of its sites held only
> 112 and 117 patients. Any number quoted from that era does not describe a
> current run.

---

## 1. The two active cohorts

Both built from **eICU** by `utils/cohort.py`, split **random** 70/10/20, task
`ehr_generation_eicu` with min 1 visit, vocabulary **pinned at 921 codes**
(fitted over all ~200 hospitals, never refitted on the cohort). 3,881
cross-hospital patients dropped in both.

| | `hilo8_random` | `lo8_random` |
|---|---:|---:|
| Built | 2026-08-19 | 2026-08-23 |
| Design | 4 sites ≥1500, 4 drawn from 100–1499 | 8 sites, **all** from 100–1499 |
| Selection seed | 0 | 1 |
| Patients | **12,156** | **3,832** |
| train / val / test | 8,509 / 1,217 / 2,430 | 2,684 / 382 / 766 |
| Size range | 202 – 2,617 | 120 – 1,301 |
| Pooled rare codes | 548 | 256 |
| Global rare (⊆ pooled) | 475 | 203 |
| Distinct codes used | 737 of 921 | 454 of 921 |
| Scoreable in **test** | **478** | **213** |
| Stays/patient (mean, p50) | 1.25, 1 | 1.36, 1 |

The two cohorts share **no hospitals**, so lo8 is an independent draw rather
than a subset.

> The p50 of 1 stay is why next-visit prediction is degenerate here — most
> patients yield no train pair.

### hilo8_random per hospital

| hospital | total | train | val | test | rare codes | rarest prevalence |
|---:|---:|---:|---:|---:|---:|---:|
| 458 | 2,617 | 1,832 | 262 | 523 | 296 | 0.0008 |
| 188 | 2,315 | 1,620 | 232 | 463 | 369 | 0.0009 |
| 300 | 2,256 | 1,579 | 226 | 451 | 302 | 0.0009 |
| 208 | 2,168 | 1,517 | 217 | 434 | 245 | 0.0009 |
| 449 | 1,430 | 1,001 | 143 | 286 | 335 | 0.0014 |
| 277 | 891 | 624 | 89 | 178 | 167 | 0.0022 |
| 358 | 277 | 194 | 28 | 55 | 104 | 0.0072 |
| 429 | 202 | 142 | 20 | 40 | 103 | 0.0099 |

**8,509** is the number to remember: every real training record across all 8
sites, and what `centralized` sees. The four large sites hold 77% of it.

### lo8_random per hospital

| hospital | total | train | val | test | rare codes | rarest prevalence |
|---:|---:|---:|---:|---:|---:|---:|
| 411 | 1,301 | 911 | 130 | 260 | 136 | 0.0015 |
| 148 | 984 | 689 | 98 | 197 | 130 | 0.0020 |
| 259 | 445 | 312 | 44 | 89 | 109 | 0.0045 |
| 396 | 403 | 282 | 40 | 81 | 113 | 0.0050 |
| 402 | 252 | 177 | 25 | 50 | 92 | 0.0079 |
| 59 | 187 | 131 | 19 | 37 | 63 | 0.0107 |
| 204 | 140 | 98 | 14 | 28 | 36 | 0.0143 |
| 123 | 120 | 84 | 12 | 24 | 51 | 0.0167 |

Rare-code density runs **opposite** to size — 0.105 codes/patient at 411 up to
0.425 at 123. The small sites are the rare-code-dense ones.

**Five of eight val folds are under 45 patients, four under 25.** Per-hospital
early stopping is not a real signal there; it falls through to `--es-fallback`.

---

## 2. What "rare" means — two nested definitions

Both come out of the cohort builder. They are **not** interchangeable.

**Rare (per-hospital)** — a code with prevalence ≤ `rare_prevalence_max` =
**0.05** at *any single hospital*, held by ≥ `rare_min_patients` = **2**
patients there. The union across the 8 sites is `pooled_rare_codes`. A code rare
at one site but routine at another *is in this set*.

**Global rare** — the subset whose prevalence across the *whole cohort* is ≤
`global_rare_prevalence_max` = **0.01**. `global_rare ⊂ pooled_rare`; the
difference is the locally-rare-but-globally-common codes, exactly where a small
site could in principle learn from a larger neighbour.

### Scope is a scoring choice, not a property

`--rare-scope hospital` (the default) gives each site *its own* tail — 296 codes
at hilo8's 458, 103 at 429. Those numbers compare to nothing. **The recipe uses
`--rare-scope pooled`** so every site is scored on one shared definition. Expect
pooled scope to *depress* the metrics, not inflate them: on hilo8, 32.7% of
scored codes had zero test positives.

**Trap:** average precision's floor *is* the prevalence, so the `global_rare`
column scores ~4× lower than `overall` for **every** arm including `prior`. That
is arithmetic, not degradation. Compare arms *within* a column, never across.

---

## 3. Generator (HALO) — `full` profile

| knob | value |
|---|---|
| `embed_dim` / `n_heads` / `n_layers` / `n_ctx` | 256 / 4 / 4 / 50 |
| `batch_size` | **128** (`sweeps/b128.yaml`; the profile default of 256 is not what runs used) |
| `lr` | 1e-4 |
| optimizer | **Adam**, PyTorch defaults, no weight decay, no schedule |
| `n_rounds` / `local_epochs` | 50 / 2 |
| `ft_epochs` | 2 (defaults to `local_epochs`) |
| FedAvg weighting | `sample` (client weight ∝ its train size) |

**Compute is matched across regimes.** FedAvg does 50 × 2 = **100** local passes
per client; `centralized` and `local` are built with `epochs` set to that same
product. Nobody wins on raw compute.

| regime | generators | trained on | passes |
|---|---|---|---|
| `centralized` | 1 | pooled real | 100 epochs |
| `fedavg` | 1 | 8 clients, averaged each round | 50 × 2 |
| `fedavg_ft` | 8 | final FedAvg global, then local | 50 × 2, + `ft_epochs` |
| `local` | 8 | each site's own train split | 100 epochs each |

**Adam state is discarded constantly.** `train_model` builds a fresh optimizer
per call — 400 over a 50-round FedAvg — and fine-tuning starts clean too. That
is standard FedAvg (no FedOpt here), but at a site taking one optimizer step per
epoch the moment estimates never leave bias-correction territory.

`val_dataset` is always passed as `None` to `train_model`, deliberately: it
would checkpoint to a `save_dir` all 8 clients share, and client B would
warm-start from client A instead of the global. Validation happens externally
via `on_epoch_end`.

---

## 4. Synthetic generation

| knob | smoke (`main.py all`) | **recipe** (`score_cohort.sh`) |
|---|---:|---:|
| `synth_per_hospital` | 2,000 | **8,000** |
| `--synth-cap` (test 1) | 0 (uncapped) | **8,000** |
| `--train-budget` (test 2) | 0 (uncapped) | **8,000** |
| `--fold` | test | test |

**8,000 is held fixed across cohorts on purpose** — it is not derived from a
cohort's own sizes. lo8 uses it even though its largest site holds 1,301
patients, so both cohorts share one prevalence resolution (1/8000) and one
classifier volume. See [RECIPE.md](RECIPE.md) stage 3.

**Uncapped scoring is not a comparison between arms.** `fedavg` and
`centralized` have no per-site ceiling, so they hold 8× what `local` and
`fedavg_ft` hold. On lo8 uncapped, `centralized` scored AP 0.0209 against
`real_pooled`'s 0.0192 — synthetic beating real, which is 16,000 classifier
records against 2,684.

`synthetic.json` holds `per_hospital` (for single-generator regimes all 8
entries are the *same* set), `pooled`, `pooled_proportional`, and
`shared_generator`.

---

## 5. Downstream classifiers

### Test 2 (rare-code TSTR) — `test2_rare_efficacy.py`

| knob | default | recipe |
|---|---|---|
| model | LSTM, multilabel over rare codes | same |
| `--epochs` / `--batch-size` | 10 / 64 | same |
| `--embedding-dim` / `--hidden-dim` | 128 / 128 | same |
| `--lr` | 1e-3 | same |
| `--mask-folds` | 4 (census over all scoreable codes) | 4 |
| `--n-eval` | 0 | 0 — the 30-code draw mode is a *separate* condition |
| `--recall-at` | 5, 10, 20 | same |
| `--min-positives` | 1 | 1 |
| `--train-budget` | 0 (uncapped) | **8,000** |
| `--fold` | val | **test** |

Classifiers are trained **centrally within each arm** — there is no federation
downstream. One arm = one classifier.

| arm family | classifiers | each trains on |
|---|---:|---|
| `centralized`, `fedavg` | 1 | the single generator's output |
| `local`, `fedavg_ft` | 8 | that hospital's own synthetic set |
| `real_local` | 8 | that hospital's **real** train split |
| `real_pooled` | 1 | all 8 real train splits — the TRTR ceiling |
| `real_pooled_budgeted` | 1 | the same, capped to `--train-budget` |
| `prior` | 0 | ranks codes by train frequency, ignores the patient |

**Test 2 reports at both levels in one run:** the pooled arms are the cohort
view, and each arm's `own_site` block is that site scored against its own val
rows. A consumer reading only the top-level numbers sees the cohort view alone.

**4-fold, not 30-draw, is what the dashboards use.** `--mask-folds 4` partitions
all scoreable codes into four folds and scores every one. `--n-eval 30` samples
30 codes with three seeds and is a separate, narrower condition
(`test2_draw30_*.json`). Don't mix them in one comparison.

### Test 1 / `eval.py` evaluator LSTM

`embed_dim` 64, `hidden_dim` 64, `batch_size` 64, `epochs` 10;
`eval_sample_cap` 200, `eval_n_bootstraps` 10, `eval_n_runs` 5.

---

## 6. Prevalence fidelity — the two targets

Test 1 scores each arm's synthetic code prevalences against a *real* target, and
**which target changes the ranking**. Both are always run:

| `--real-scope` | target | question |
|---|---|---|
| `hospital` | each site's own test fold (24–523 patients) | is it a good **specialist**? |
| `pooled` | the whole cohort test fold (766 / 2,430) | is it a good **generalist**? |

Reporting one alone presents a choice of target as a fact about the generators.
The dashboard's head-vs-tail panel switches between them.

Metrics: Pearson, R² (against the identity line, so it can go very negative),
RMSE — each over all codes and over rare codes separately.

---

## 7. Output filenames

The test scripts' default `--out` is **cohort-agnostic**, so a run on any cohort
claims `test1_prevalence.json` / `test2_rare_efficacy.json`. Scoring lo8 once
silently replaced the hilo8 results the dashboards were built from — same shape,
no error. `main.py` now derives `--out` from the cohort cache name and
`score_cohort.sh` names every file for the cohort. **Never rely on the bare
default.** A variant condition additionally needs `SUFFIX=` (e.g. `_rw`).

---

## 8. Provenance

- `hilo8_random` manifest built 2026-08-19, git `efec897`; `lo8_random`
  2026-08-23
- Cohort cache and eICU root come from `$FEDCOHORT_CACHE` / `$EICU_ROOT`; paths
  are deliberately not recorded here
- Regenerate from `manifest.json` (cohort stats), `PROFILES["full"]` in
  `train.py` (generator knobs), `_build_arg_parser` in `test2_rare_efficacy.py`
  (classifier knobs), and `<save_dir>/config.json` (what a run actually used)

Contains cohort-level aggregates only — no patient-level rows, no paths, no
credentials.
