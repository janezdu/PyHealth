# eICU hospital sizes

Where the cohort's hospitals come from, and what is available if it is
rebuilt. Generated from `patient.csv` by counting **unique patients**
(`patienthealthsystemstayid`), not unit stays -- a patient can have several.

Aggregate counts only: no patient-level data, and eICU hospital IDs are
already de-identified in the source.

## The dataset

| | |
|---|---:|
| hospitals | 208 |
| unique patients | 166,355 |
| unit stays | 200,859 |
| largest hospital | 5,509 |
| smallest hospital | 3 |
| median hospital | 473 |
| mean hospital | 799 |

## Size distribution

| band | hospitals | patients | share of dataset |
|---|---:|---:|---:|
| 0-99 | 33 | 1,183 | 0.7% |
| 100-199 | 17 | 2,392 | 1.4% |
| 200-499 | 59 | 19,988 | 12.0% |
| 500-999 | 42 | 29,883 | 18.0% |
| 1000-1499 | 25 | 30,284 | 18.2% |
| 1500-2999 | 20 | 38,458 | 23.1% |
| 3000+ | 12 | 44,167 | 26.5% |

The distribution is steeply skewed: the 32 hospitals at 1500+ hold
50% of all patients, and the 176 below 1500 hold the rest.

## Percentiles

| percentile | patients |
|---|---:|
| p99 | 3,872 |
| p95 | 3,102 |
| p90 | 1,793 |
| p75 | 1,067 |
| p50 | 473 |
| p25 | 213 |
| p10 | 47 |

## The 20 largest

| hospital | patients | ~post-task |
|---|---:|---:|
| 73 | 5,509 | ~4,271 |
| 264 | 4,707 | ~3,649 |
| 420 | 3,876 | ~3,005 |
| 338 | 3,831 | ~2,970 |
| 243 | 3,608 | ~2,797 |
| 167 | 3,370 | ~2,612 |
| 458 | 3,320 | ~2,573 |
| 300 | 3,306 | ~2,563 |
| 188 | 3,266 | ~2,532 |
| 443 | 3,208 | ~2,487 |
| 208 | 3,147 | ~2,439 |
| 252 | 3,019 | ~2,340 |
| 199 | 2,727 | ~2,114 |
| 122 | 2,709 | ~2,100 |
| 176 | 2,440 | ~1,891 |
| 281 | 2,269 | ~1,759 |
| 411 | 2,153 | ~1,669 |
| 413 | 2,100 | ~1,628 |
| 449 | 1,990 | ~1,542 |
| 283 | 1,861 | ~1,442 |

`~post-task` scales by 0.78, the shrinkage observed on hospital 420 in the
current cohort (3005 kept of 3876 raw) once `MIN_VISITS` and the
cross-hospital-patient drop are applied. Treat it as an estimate.

## Pools for a 4-big / 4-small cohort

- **>= 1500 patients:** 32 hospitals to choose 4 from
- **< 1500 patients:** 176 hospitals to choose 4 from

Taking the largest four of the big band gives ~13,900 post-task patients
on its own -- already 1.7x the entire current cohort (8,282).

Note `draw_bands` in `utils/cohort.py` samples *uniformly at random* within
a band, by design, so that results generalise beyond the biggest sites.
Maximising total size means passing `--hospitals` explicitly instead.

## The current cohort in context

| hospital | patients | percentile in eICU |
|---|---:|---:|
| 420 | 3,876 | p99 |
| 199 | 2,727 | p94 |
| 345 | 1,645 | p87 |
| 79 | 1,213 | p79 |
| 259 | 553 | p55 |
| 253 | 438 | p47 |
| 438 | 190 | p24 |
| 201 | 133 | p19 |

Six of the eight sit above eICU's median hospital (473 patients); only 438
(p24) and 201 (p19) are genuinely small by dataset standards. The cohort is
therefore tilted large, which is worth remembering when reading a result as
being about "small hospitals".

---

Regenerate with `python examples/fedpyhealth/utils/eicu_sizes.py`
(needs `EICU_ROOT`). Numbers here were generated 2026-08-19.


---

hospital  patients  rare-set possible
  ------------------------------------
       264     3,529  yes
       420     3,005  yes
       243     2,723  yes
       338     2,671  yes
       458     2,617  yes
       443     2,432  yes
        73     2,423  yes
       188     2,315  yes
       300     2,256  yes
       252     2,223  yes
       199     2,218  yes
       208     2,168  yes
       167     2,161  yes
       122     2,101  yes
       176     1,667  yes
       281     1,522  yes
       394     1,477  yes
       416     1,476  yes
       449     1,430  yes
       417     1,328  yes
       411     1,301  yes
       142     1,257  yes
       283     1,250  yes
       197     1,249  yes
       307     1,212  yes
       110     1,209  yes
       165     1,173  yes
       248     1,173  yes
       365     1,161  yes
       183     1,124  yes
       331     1,124  yes
       345     1,122  yes
       400     1,119  yes
       435     1,061  yes
       227     1,042  yes
       141     1,010  yes
       148       984  yes
        79       972  yes
       171       932  yes
       444       927  yes
       413       894  yes
       277       891  yes
       226       883  yes
       440       851  yes
       318       849  yes
       403       821  yes
       157       816  yes
       195       807  yes
       217       801  yes
       390       791  yes
       388       786  yes
       154       784  yes
       384       763  yes
       244       731  yes
       382       699  yes
       452       683  yes
       202       654  yes
       181       650  yes
       271       647  yes
       280       646  yes
       310       644  yes
       272       630  yes
       336       629  yes
       391       622  yes
       206       601  yes
       146       588  yes
       301       587  yes
       144       586  yes
       357       575  yes
       184       567  yes
       220       556  yes
       282       528  yes
       353       524  yes
       143       521  yes
       392       511  yes
       215       493  yes
        63       482  yes
       407       467  yes
       198       460  yes
       269       460  yes
       279       453  yes
       259       445  yes
       459       442  yes
       152       439  yes
       256       429  yes
       386       412  yes
       275       409  yes
       396       403  yes
       312       401  yes
        92       400  yes
       140       399  yes
       421       395  yes
       383       391  yes
       419       389  yes
       268       383  yes
       210       376  yes
       200       371  yes
       337       368  yes
       397       361  yes
       245       346  yes
       224       342  yes
        71       333  yes
       387       331  yes
       389       329  yes
        66       315  yes
       253       291  yes
       175       290  yes
       254       288  yes
       180       282  yes
       194       282  yes
       358       277  yes
       196       272  yes
       328       270  yes
       112       268  yes
       155       267  yes
       424       267  yes
       405       265  yes
       364       262  yes
       404       262  yes
       434       258  yes
       205       257  yes
       108       256  yes
       402       252  yes
       436       249  yes
       249       238  yes
       182       236  yes
       360       228  yes
       439       227  yes
       398       223  yes
       445       218  yes
       258       209  yes
       412       206  yes
       251       203  yes
       429       202  yes
        69       200  yes
        59       187  yes
       399       185  yes
        95       184  yes
       158       181  yes
        68       181  yes
       266       177  yes
       133       172  yes
       342       171  yes
       267       163  yes
       381       163  yes
       422       156  yes
       250       155  yes
       207       152  yes
       408       151  yes
       125       143  yes
       204       140  yes
        67       128  yes
       138       126  yes
       262       123  yes
       123       120  yes
       201       117  yes
       303       112  yes
       438       112  yes
        60       111  yes
        58       105  yes
       164       104  yes
       203       100  yes
       433        97  yes
       273        95  yes
        56        91  yes
       393        90  yes
       120        86  yes
       246        82  yes
       428        72  yes
       425        69  yes
       355        66  yes
       209        65  yes
        61        65  yes
       174        57  yes
       131        55  yes
       350        49  yes
       363        49  yes
        93        49  yes
       263        48  yes
       356        47  yes
        85        41  yes
       437        39  NO
        83        35  NO
       115        32  NO
       352        31  NO
       102        30  NO
        84        30  NO
       401        29  NO
       447        26  NO
       414        21  NO
        96        21  NO
        91        15  NO
       156        14  NO
        90        14  NO
       179        13  NO
       265        12  NO
       361        12  NO
        86        11  NO
       323         9  NO
       423         9  NO
        94         8  NO
       351         6  NO
       135         5  NO
       212         5  NO
       151         4  NO
       136         2  NO
       385         2  NO



cohort 'hilo8_random'   split=random   guaranteed=none
  8 hospitals, 737 distinct codes of 921 in vocab, 548 pooled rare
```
CODES by fold presence
  present in      rare  not rare  all
  -----------------------------------
  train+val+test   373         8  381
  train+val         39        10   49
  train+test       103        18  121
  val+test           1         2    3
  train             30       110  140
  val                1        14   15
  test               1        27   28
  TOTAL            548       189  737
  ```
  (184 more codes exist in the pinned vocabulary but no cohort patient carries them)
  NOTE: 'not rare' is not the same as common -- 125 of those 189 codes have < 2 carriers in the whole cohort
        and were excluded from the rare set by the minimum-carriers floor, not by prevalence.

  rare codes reachable per fold: train 545/548 (99.5%)   val 414/548 (75.5%)   test 478/548 (87.2%)

PATIENTS by fold
```
  hospital  train   val  test    all       fractions
  --------------------------------------------------
  458        1832   262   523   2617  0.70 0.10 0.20
  188        1620   232   463   2315  0.70 0.10 0.20
  300        1579   226   451   2256  0.70 0.10 0.20
  208        1517   217   434   2168  0.70 0.10 0.20
  449        1001   143   286   1430  0.70 0.10 0.20
  277         624    89   178    891  0.70 0.10 0.20
  358         194    28    55    277  0.70 0.10 0.20
  429         142    20    40    202  0.70 0.10 0.20
  TOTAL      8509  1217  2430  12156  0.70 0.10 0.20
```