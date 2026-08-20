# Federated synthetic-EHR experiment — cheatsheet

Reference numbers for the `strat8_random` cohort and the four training regimes.
Everything here is read off the frozen cohort manifest and the run configs; if you
rebuild the cohort or change a profile, regenerate this (see [Provenance](#provenance)).

---

## 1. Cohort

Built from **eICU** by `utils/cohort.py`. 207 hospitals in the source; 8 selected.

| | |
|---|---|
| Cohort name | `strat8_random` |
| Hospitals | 8, sampled 2 per size band |
| Size bands | `0-199`, `200-499`, `500-1999`, `2000-` |
| Selection seed | 0 |
| Split | **random**, 70 / 10 / 20 (train / val / test) |
| Task | `ehr_generation_eicu`, min 1 visit |
| Vocabulary | 921 codes (cohort actually uses 779) |
| Patients dropped | 3,881 (appeared at more than one hospital) |
| Stays per patient | mean 1.31, p50 **1**, p90 2, max 19 |

> The p50 of 1 stay is why next-visit prediction is degenerate here and why
> `metrics="privacy"` rather than `"utility"` — most patients yield no train pair.

### Per hospital

| hospital | total | train | val | test | rare codes | rarest code prevalence |
|---:|---:|---:|---:|---:|---:|---:|
| 420 | 3,005 | 2,104 | 300 | 601 | 562 | 0.0007 |
| 199 | 2,218 | 1,552 | 222 | 444 | 299 | 0.0009 |
| 345 | 1,122 | 786 | 112 | 224 | 165 | 0.0018 |
| 79 | 972 | 681 | 97 | 194 | 296 | 0.0021 |
| 259 | 445 | 312 | 44 | 89 | 109 | 0.0045 |
| 253 | 291 | 204 | 29 | 58 | 100 | 0.0069 |
| 438 | 112 | 79 | 11 | 22 | 23 | 0.0179 |
| 201 | 117 | 82 | 12 | 23 | 29 | 0.0171 |
| **total** | **8,282** | **5,800** | **827** | **1,655** | | |

**5,800** is the number to remember: it's every real training record across all 8
hospitals, and it's what the `centralized` generator sees. The largest site holds 36%
of it; the smallest holds 1.4%.

---

## 2. What "rare" means — two nested definitions

Both come out of the cohort builder. They are **not** interchangeable.

### Rare (per-hospital) → 641 codes

A code is rare if, **at any single hospital**, it has

- prevalence ≤ `rare_prevalence_max` = **0.05**, and
- at least `rare_min_patients` = **2** patients

The manifest states the scope explicitly:
`rare_scope: "per-hospital: rare at any one hospital counts as rare"`.

The union across all 8 hospitals is `pooled_rare_codes` — **641 codes**. A code that is
rare at hospital 438 but routine at hospital 420 *is in this set*.

### Global rare → 513 of those 641

Of the 641, the subset whose prevalence **across the whole cohort** is ≤
`global_rare_prevalence_max` = **0.01**. Computed in `utils/cohort.py` from the summed
per-fold support divided by 8,282 patients.

`global_rare ⊂ pooled_rare`. The ~128-code difference is the locally-rare-but-globally-common
codes — the ones where a small site could in principle learn from a larger neighbour.
Global rare is the genuinely-rare-everywhere set.

### What Test 2 actually scores

Test 2 only scores codes with at least `--min-positives` (default **1**) positives in
the evaluation fold, so the pools shrink:

| fold | codes with ≥1 positive |
|---|---:|
| train | 637 |
| **val** (default) | **476** |
| test | 551 |

So on the default `--fold val` run:

| column in the results table | codes |
|---|---:|
| `overall` / "all rare" | **476** |
| `global_rare` | **348** |

**Trap:** average precision's floor *is* the prevalence, so the `global_rare` column
scores ~4× lower than `overall` for **every** arm, including `prior`. That is arithmetic,
not degradation. Compare arms *within* a column, never across.

---

## 3. Generator (HALO) — `full` profile

| knob | value |
|---|---|
| `embed_dim` | 256 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `n_ctx` | 50 |
| `batch_size` | 64 |
| `lr` | 1e-4 |
| `n_rounds` | **50** |
| `local_epochs` | **2** |
| `ft_epochs` | 2 (defaults to `local_epochs`) |
| FedAvg weighting | `sample` (client weight ∝ its train size) |

### Compute budget is matched across regimes

FedAvg does `n_rounds × local_epochs` = **50 × 2 = 100** local passes over each client's
data. The non-federated baselines are built with `epochs` set to that same product, so
`centralized` and `local` each get 100 epochs. Nobody wins on raw compute.

### What each regime is

| regime | generators | trained on | passes |
|---|---|---|---|
| `centralized` | 1 | pooled real, 5,800 records | 100 epochs |
| `fedavg` | 1 | 8 clients, averaged each round | 50 rounds × 2 |
| `fedavg_ft` | 8 | final FedAvg global, then 2 local epochs per site | 50 × 2, + 2 |
| `local` | 8 | each site's own train split only | 100 epochs each |

---

## 4. Synthetic generation

| knob | value |
|---|---|
| `synth_per_hospital` | **2,000** — every site generates this many regardless of its real size |
| `num_synth` | 5,000 — size of the *pooled* view only |

At 2,000, hospital 438 gets **25×** the synthetic data it holds real. That ratio is the
claim under test.

`synthetic.json` holds four keys:

| key | what it is |
|---|---|
| `per_hospital` | `{hid: [2000 patients]}` — for single-generator regimes all 8 entries are the *same* set |
| `pooled` | uniform mix, subsampled to `num_synth` |
| `pooled_proportional` | mixed ∝ real hospital size, subsampled to `num_synth` |
| `shared_generator` | bool — true for `centralized` / `fedavg` |

**Resolution caveat:** a set of N synthetic patients resolves prevalence only to 1/N.
At 2,000 that's 5e-4 — comfortably below hospital 438's rarest code (0.0179) but *above*
hospital 420's deepest tail (6.7e-4). Band Test 1's rare metrics by support before
claiming anything about the deep tail.

---

## 5. Downstream classifiers

### Test 2 (rare-code TSTR) — `test2_rare_efficacy.py`

| knob | default |
|---|---|
| model | LSTM, multilabel over rare codes |
| `--epochs` | 10 |
| `--batch-size` | 64 |
| `--embedding-dim` | 128 |
| `--hidden-dim` | 128 |
| `--lr` | 1e-3 |
| `--mask-folds` | 4 |
| `--recall-at` | 5, 10, 20 |
| `--min-positives` | 1 |
| `--train-budget` | 0 (uncapped) |
| `--fold` | val |

Classifiers are trained **centrally within each arm** — there is no federation
downstream. One arm = one classifier.

| arm family | classifiers | each trains on |
|---|---:|---|
| `centralized`, `fedavg` | 1 | the single generator's output |
| `local`, `fedavg_ft` | 8 | that hospital's own synthetic set |
| `real_local` | 8 | that hospital's **real** train split |
| `real_pooled` | 1 | all 8 real train splits (5,800) — the TRTR ceiling |
| `prior` | 0 | ranks codes by train frequency, ignores the patient |

### Test 1 / `eval.py` evaluator LSTM

`embed_dim` 64, `hidden_dim` 64, `batch_size` 64, `epochs` 10;
`eval_sample_cap` 200, `eval_n_bootstraps` 10, `eval_n_runs` 5.

---

## 6. Baselines and floors

| baseline | AUROC | AP · all rare | recall@10 |
|---|---:|---:|---:|
| chance | 0.500 | = prevalence | — |
| `prior` | 0.500 | 0.0147 | 0.544 |
| `real_local` | 0.377 | 0.0176 | 0.280 |
| `real_pooled` | 0.711 | 0.0621 | 0.620 |

- **AUROC** has a fixed chance floor of 0.500 by construction. Below it means
  anti-correlated, not weak.
- **AP** has no fixed floor — its floor is the code's prevalence, which is why the
  `global_rare` column sits so much lower.
- **recall@k** has no analytic floor either; use the `prior` row.

---

## 7. Provenance

- Cohort manifest built 2026-08-11, git `dff22aa`
- Cohort cache and eICU root are configured via `--cohort-cache` / `--eicu-root`;
  paths are deliberately not recorded here
- Regenerate these numbers from `manifest.json` (cohort stats), `PROFILES["full"]`
  in `train.py` (generator knobs), `_build_arg_parser` in `test2_rare_efficacy.py`
  (classifier knobs), and `<save_dir>/config.json` (what a given run actually used)

Contains cohort-level aggregates only — no patient-level rows, no paths, no credentials.
