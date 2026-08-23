# eICU cohort `hilo8_random` — data distributions

12,156 patients across 8 hospitals, all folds combined. Vocabulary 921, 737 distinct codes present.

## Trajectory length and codes per patient

| hosp | n_pat | visits/pt mean | p50 | p90 | max | distinct codes/pt mean | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 458 | 2617 | 1.22 | 1 | 2 | 12 | 2.75 | 2 | 5 | 10 | 18 |
| 188 | 2315 | 1.24 | 1 | 2 | 8 | 7.18 | 6 | 14 | 24 | 34 |
| 300 | 2256 | 1.25 | 1 | 2 | 9 | 2.05 | 1 | 4 | 7 | 12 |
| 208 | 2168 | 1.36 | 1 | 2 | 22 | 2.41 | 2 | 5 | 9 | 13 |
| 449 | 1430 | 1.17 | 1 | 2 | 6 | 3.07 | 2 | 6 | 13 | 21 |
| 277 | 891 | 1.34 | 1 | 2 | 10 | 3.06 | 2 | 6 | 12 | 17 |
| 358 | 277 | 1.10 | 1 | 1 | 3 | 4.02 | 3 | 8 | 13 | 18 |
| 429 | 202 | 1.25 | 1 | 2 | 4 | 8.89 | 8 | 15 | 25 | 27 |
| ALL | 12156 | 1.25 | 1 | 2 | 22 | 3.59 | 2 | 8 | 17 | 34 |

> Median patient has **one visit** at every hospital. 90th percentile is 2. This is an ICU cohort of mostly single unit-stays, which is why anything conditioned on visit history has little to condition on.

## Codes per visit, by visit position

| hosp | visit 1 mean | (n) | visit 2 mean | (n) | visit 3 mean | (n) |
|---|---:|---:|---:|---:|---:|---:|
| 458 | 10.37 | 2617 | 14.11 | 431 | 12.77 | 93 |
| 188 | 31.50 | 2315 | 40.19 | 383 | 52.84 | 105 |
| 300 | 2.15 | 2256 | 2.25 | 397 | 2.87 | 103 |
| 208 | 4.30 | 2168 | 5.41 | 473 | 4.46 | 142 |
| 449 | 13.40 | 1430 | 17.39 | 188 | 22.55 | 33 |
| 277 | 5.58 | 891 | 6.85 | 196 | 9.77 | 62 |
| 358 | 10.24 | 277 | 12.69 | 26 | 53.00 | 2 |
| 429 | 29.14 | 202 | 27.94 | 36 | 13.55 | 11 |
| ALL | 12.10 | 12156 | 14.49 | 2130 | 16.83 | 551 |

> Later visits are **larger**, cohort-wide 12.10 → 14.49 → 16.83. That is a selection effect — patients who return are sicker — not a property of visits. Every HALO arm trained here reproduces the opposite sign.

## Long tail, per hospital

| hosp | n_pat | distinct codes | top code prev | top-10 sum | median prev | ≤1% | ≤0.5% | gini | codes for 50% of mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 458 | 2617 | 441 | 0.208 | 0.96 | 0.0011 | 87% | 78% | 0.778 | 19 |
| 188 | 2315 | 517 | 0.462 | 2.24 | 0.0022 | 76% | 67% | 0.794 | 25 |
| 300 | 2256 | 443 | 0.118 | 0.61 | 0.0013 | 90% | 81% | 0.716 | 27 |
| 208 | 2168 | 377 | 0.201 | 0.99 | 0.0014 | 87% | 79% | 0.773 | 16 |
| 449 | 1430 | 491 | 0.113 | 0.80 | 0.0021 | 87% | 77% | 0.694 | 32 |
| 277 | 891 | 289 | 0.201 | 1.13 | 0.0022 | 79% | 65% | 0.724 | 17 |
| 358 | 277 | 212 | 0.184 | 1.27 | 0.0072 | 56% | 41% | 0.613 | 22 |
| 429 | 202 | 225 | 0.574 | 3.04 | 0.0149 | 48% | 34% | 0.661 | 20 |

> `gini` 0 = all codes equally common, 1 = all mass on one code. `codes for 50% of mass` = how many of the most common codes account for half of all code occurrences.

### What stands out

- **188 is the documentation outlier**: 31.5 codes in a first visit against 300's 2.15, a 15x range. Most distinct codes (517), highest top-10 mass (2.24 — patients average 2.24 of the top 10 codes each). Same median age and sex mix as everyone else, so this is coding practice, not case mix.
- **429 is the most concentrated**: its top code appears in **57.4%** of patients and its median code prevalence (0.0149) is 13x hospital 458's. With 202 patients its tail is shallow — only 48% of its codes fall below 1% prevalence, against 87–90% at the large sites.
- **The small sites have shallower tails by construction.** A code cannot have prevalence below 1/n, so 358 (277 patients) and 429 (202) cannot express anything under 0.0036 and 0.0050. Their lower `≤1%` percentages are a resolution floor, not a different disease mix.
- **Heterogeneity here is documentation, not demographics.** Codes-per-visit spans 2.15 to 31.50 across sites while median age spans 59–67 and female share 41–53%.
