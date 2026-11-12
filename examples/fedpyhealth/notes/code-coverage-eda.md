# EDA plan: does federation put codes in a hospital's mouth that it could not say alone?

**Status:** planned, not implemented. Nothing here has been run.

## The question

A hospital's own generator can only emit codes it saw. Federation is supposed to
lend it the rest of the cohort's vocabulary. That claim has never been measured
directly — every result so far is a *prevalence* comparison, which scores how
well the emitted distribution matches and says nothing about which codes were
reachable at all.

The specific measurement, for each hospital `h`:

> Of the codes that `h`'s **local-only** generator never emits, how many does the
> **fedavg** global generator emit? How many does **fedavg_ft** (that hospital's
> fine-tuned copy) emit? And of those, how many does `h` actually *need* — i.e.
> appear in `h`'s own held-out real data?

The last clause is what turns a vocabulary statistic into a utility one. A
federated model that emits 900 codes trivially "covers" everything local missed;
that is hedging, not sharing.

## Why this is worth doing separately from Test 1 and Test 2

Test 1 correlates prevalences over codes both sides emit. Test 2 trains a
downstream classifier. Neither can distinguish these two situations:

* local misses code `c` entirely; fedavg emits it at roughly the right rate
  — *federation supplied a code*;
* local misses `c`; fedavg emits it at 50× its real rate
  — *federation supplied noise that happens to include `c`*.

Both improve a Pearson correlation. Only the second is worthless, and the split
is exactly what the rare-code claim in the paper rests on.

## Sets, defined once

For hospital `h`, over the scored code pool:

| symbol | definition |
|---|---|
| `R_h` | codes in `h`'s **real test** split — what `h` actually sees |
| `T_h` | codes in `h`'s **real train** split — what a local model could have learned |
| `L_h` | codes emitted by **local-only**'s generator for `h` |
| `F`   | codes emitted by the **fedavg** global generator (one set, all sites) |
| `Fh`  | codes emitted by **fedavg_ft**'s generator for `h` |
| `C`   | codes emitted by **centralized** |

The headline quantities, per hospital:

```
missed_h      = R_h \ L_h              # needed, local could not say it
recovered_fed = |missed_h ∩ F|  / |missed_h|
recovered_ft  = |missed_h ∩ Fh| / |missed_h|
recovered_cen = |missed_h ∩ C|  / |missed_h|
```

and the discipline term that stops a hedging model from winning:

```
spurious_fed  = |F \ R_h| / |F|        # emitted, h has no use for it
```

Report `recovered_*` against `spurious_*` as a pair. A model that emits the whole
vocabulary scores `recovered = 1.0` and `spurious ≈ 0.95`, and the pair says so
immediately where either number alone would not.

## Three cuts that matter

1. **By hospital size.** 429 has 142 train patients and 358 has 179; 458 has
   1832. If federation helps anyone it is the small sites, and `missed_h` should
   be much larger for them. This is the single most informative split.
2. **By code rarity.** Split on the manifest's `global_rare_codes` (cohort
   prevalence ≤1%). Recovery of common codes is uninteresting — local sees them
   too. The claim lives entirely in the rare half.
3. **Local-alone vs local-ever.** `R_h \ T_h` is the part `h` could not have
   learned even in principle, because it is absent from its own training split.
   That subset is the honest target: federation cannot be credited for a code
   the hospital already had.

## Controls, without which the numbers mean nothing

* **Emission budget.** Every arm must be sampled to the same number of synthetic
  patients before sets are compared. Coverage grows monotonically with samples
  drawn, so an arm allowed more patients wins for free. Use the existing
  `--synth-per-hospital 8000`.
* **A real-data ceiling.** Compute `recovered` using the *pooled real train* set
  in place of a generator. That is the best any federated method could do, and
  turns every figure into a fraction of achievable rather than an absolute.
* **A prior floor.** Draw the same number of patients from the pooled marginal
  code distribution, independently per code. This floor is high and that is the
  point — it is what "emitting a lot" alone buys.
* **Frequency threshold.** "Emits code `c`" should mean *at least k patients
  carry it*, not "at least one", or a single hallucinated code counts as
  coverage. Report at k ∈ {1, 5} and expect them to disagree for the hedging arms.

## Implementation sketch

New script, `examples/fedpyhealth/scripts/eda_code_coverage.py`, CPU-only,
minutes not hours. It needs no model: every arm's `synthetic.json` is already on
disk, and `load_synthetic` / `read_trajectories` / `scored_codes` in
`utils/cohort.py` already return exactly the structures required.

```
for each arm in {local, fedavg, fedavg_ft, centralized, real_pooled, prior}:
    load synthetic.json          -> {hospital: [patient -> visits]}
    cap to the emission budget
    per hospital: set of codes with >= k carriers
for each hospital:
    R_h, T_h from read_trajectories(cache, "test"/"train", [h])
    emit the table above, split by rare / common
```

Output: one JSON plus a small-multiples figure — eight hospitals on the x axis
ordered by train size, `recovered_*` as grouped bars, `spurious_*` as a marker
on the same axis.

## What would count as a result

The claim the project wants is: **`recovered_fed` is high and rises as hospital
size falls, while `spurious_fed` stays near the real-data ceiling's value.**

The likely spoiler, given what we already know: HALO's sigmoid decoder emits
around 460 codes/visit when it hedges, so `F` may be close to the whole
vocabulary and `recovered_fed ≈ 1.0` for trivial reasons. If that happens the
k ≥ 5 threshold and `spurious_fed` are what save the analysis — and the finding
becomes "federation covers everything because it emits everything", which is a
real and reportable negative.

Related: [[adapters]], `notes/README.md`.
