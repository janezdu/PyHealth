# Prevalence fidelity — all codes vs rare codes

Source: `_outputs/results/tests/test1_prevalence_pooledreal_pooledrare.json`  
Cohort `hilo8_random`, fold `test`, `--real-scope pooled` · `--rare-scope pooled` · `--synth-cap 8000`

Every hospital's synthetic set (8,000 patients) is scored against the **same 2,430-patient pooled test fold**, over the **same 478 rare codes** (the 548-code union of all sites' rare sets, intersected with the pooled fold). So all arms and all sites face one identical target and one identical code set.

> **Read Pearson and RMSE; treat R² as directional.** Measured from eight byte-identical `fedavg` comparisons, the unseeded bootstrap in `compute_prevalence_metrics` gives R² a spread of 0.105 (sd 0.031) when healthy and 16.6 (sd 6.2) for `centralized`, against 0.010 (sd 0.003) for Pearson.

## Summary — median across the 8 hospitals

| metric | arm | all codes | rare codes | gap (rare − all) |
|---|---|---:|---:|---:|
| Pearson | `local` | 0.871 | 0.805 | -0.066 |
| Pearson | `fedavg_ft` | 0.945 | 0.938 | -0.006 |
| Pearson | `fedavg` | 0.977 | 0.957 | -0.020 |
| Pearson | `centralized` | 0.710 | 0.643 | -0.068 |
| R² | `local` | 0.574 | 0.380 | -0.194 |
| R² | `fedavg_ft` | 0.491 | 0.258 | -0.233 |
| R² | `fedavg` | 0.684 | 0.631 | -0.053 |
| R² | `centralized` | -37.404 | -64.926 | -27.522 |
| RMSE | `local` | 0.0088 | 0.0104 | 0.0016 |
| RMSE | `fedavg_ft` | 0.0092 | 0.0121 | 0.0029 |
| RMSE | `fedavg` | 0.0077 | 0.0084 | 0.0008 |
| RMSE | `centralized` | 0.0869 | 0.1070 | 0.0200 |

## Per hospital — `local`

| hospital | n_train | Pearson all | Pearson rare | Pearson gap | R² all | R² rare | RMSE all | RMSE rare |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 458 | 1832 | 0.904 | 0.877 | -0.027 | 0.737 | 0.653 | 0.0076 | 0.0082 |
| 188 | 1620 | 0.910 | 0.839 | -0.071 | -2.665 | -3.599 | 0.0260 | 0.0246 |
| 300 | 1579 | 0.887 | 0.802 | -0.086 | 0.779 | 0.618 | 0.0069 | 0.0091 |
| 208 | 1517 | 0.878 | 0.842 | -0.036 | 0.731 | 0.632 | 0.0077 | 0.0082 |
| 449 | 1001 | 0.864 | 0.808 | -0.057 | 0.722 | 0.595 | 0.0071 | 0.0084 |
| 277 | 624 | 0.798 | 0.785 | -0.013 | 0.425 | 0.164 | 0.0100 | 0.0116 |
| 358 | 194 | 0.842 | 0.793 | -0.049 | 0.116 | -0.689 | 0.0118 | 0.0154 |
| 429 | 142 | 0.822 | 0.714 | -0.108 | -5.447 | -6.397 | 0.0314 | 0.0380 |

## Per hospital — `fedavg_ft`

| hospital | n_train | Pearson all | Pearson rare | Pearson gap | R² all | R² rare | RMSE all | RMSE rare |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 458 | 1832 | 0.932 | 0.897 | -0.034 | 0.617 | 0.587 | 0.0091 | 0.0093 |
| 188 | 1620 | 0.972 | 0.961 | -0.011 | -2.455 | -4.195 | 0.0285 | 0.0290 |
| 300 | 1579 | 0.934 | 0.900 | -0.033 | 0.864 | 0.796 | 0.0052 | 0.0065 |
| 208 | 1517 | 0.918 | 0.885 | -0.032 | 0.748 | 0.611 | 0.0082 | 0.0077 |
| 449 | 1001 | 0.940 | 0.933 | -0.007 | 0.441 | 0.249 | 0.0088 | 0.0113 |
| 277 | 624 | 0.970 | 0.949 | -0.020 | 0.541 | 0.267 | 0.0093 | 0.0128 |
| 358 | 194 | 0.969 | 0.948 | -0.021 | 0.293 | -0.082 | 0.0110 | 0.0129 |
| 429 | 142 | 0.949 | 0.944 | -0.006 | -2.002 | -4.224 | 0.0232 | 0.0290 |

## Notes

- `fedavg` and `centralized` share one generator, so their eight per-hospital rows are byte-identical comparisons; their spread is bootstrap noise, not per-site variation, and only the median is meaningful. `local` and `fedavg_ft` have genuine per-site generators.
- `fedavg_ft` beats `local` on Pearson at **all 8 hospitals**, on both code sets. The margin is largest at the smallest sites: at 429 (142 train patients) rare Pearson goes 0.714 → 0.944 and the head-to-tail gap collapses from −0.108 to −0.006.
- Against each site's **own** test fold instead of the pooled one, `local` wins. The two targets ask different questions — "does this generator match its own site" vs "does it match the cohort" — and each answer is correct for its question.
- `centralized` is broken on every metric: it emits a median of 26 codes per visit against a real median of 4, with 6 codes appearing in over half its patients (real prevalence never exceeds 0.19).
- Hospitals 188 and 429 show strongly negative R² alongside the *highest* Pearson for both arms — shape right, scale off. Unexplained.
