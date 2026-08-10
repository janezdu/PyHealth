"""Freeze the federated-recoverable tail band and its per-hospital val split.

The band is the set of **(hospital, code) pairs** where federation is the only
thing that can help:

  * the hospital has >= ``--val-quota`` patients carrying the code  -> enough
    held-out positives to MEASURE the code at that hospital;
  * after holding those out, the hospital has <= ``--local-train-max`` left  ->
    too few for that hospital's own generator to LEARN it alone;
  * the cohort as a whole still has >= ``--global-train-min``  -> enough for a
    federated generator to learn it.

Pairs, not codes, are the unit: a code can be plentiful at one hospital (which
can learn it unaided) and scarce at three others (which cannot).  Those three
(hospital, code) pairs are exactly where a federated generator should beat a
local one, and scoring them per hospital is what makes the difference visible --
a POOLED comparison hides it, because the one hospital that owns the code
generates it and it lands in the pooled synthetic either way.

The val split is quota-based, not random: a random 20% holdout leaves many band
pairs with 2-3 positives, which is too few to score and makes band membership
depend on the seed.  Codes are filled rarest-first, and patients already picked
for one code count toward every other code they carry, so the quota is met with
far fewer held-out patients than quota x codes.  Within a code the pick is
uniform among that hospital's carriers, so per-code positives stay an unbiased
sample rather than being skewed toward multi-morbid patients.

Outputs a frozen manifest (reuse it byte-identically across every training
regime -- if fedavg and local re-split, the comparison is meaningless) and a CSV
of per-hospital, per-code patient and occurrence counts on both sides of the
split.

Reads the hospital_stats.py cache (needs cache_version >= 3, which carries
per-patient occurrence counts).  Runs in seconds; no eICU load, no GPU::

    python examples/fedpyhealth/select_tail_band.py \
        --cache _outputs/eda/hospital_stats_cache.json --n-clients 8
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

REQUIRED_CACHE_VERSION = 3


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--cache", default="_outputs/eda/hospital_stats_cache.json",
                   help="hospital_stats.py cache (needs cache_version >= 3)")
    p.add_argument("--n-clients", type=int, default=8,
                   help="cohort size: the largest N hospitals (default 8)")
    p.add_argument("--min-hospital-records", type=int, default=500,
                   help="only consider hospitals with >= this many records")
    p.add_argument("--cohort-file",
                   help="JSON manifest of hospitals to use instead of largest-N")
    p.add_argument("--cap-per-hospital", type=int, default=None,
                   help="use at most this many patients per hospital (random "
                        "sample at --seed). FedAvg cost scales with total data, "
                        "so this is the main compute dial: it shrinks the band "
                        "too, so check the printed pair count. Default: no cap.")
    p.add_argument("--val-quota", type=int, default=10,
                   help="held-out patients per band pair (default 10)")
    p.add_argument("--local-train-max", type=int, default=10,
                   help="max patients left at the hospital after the holdout for "
                        "the pair to still count as locally unlearnable")
    p.add_argument("--global-train-min", type=int, default=30,
                   help="min cohort-wide training patients for the pair to be "
                        "federated-learnable")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="after quotas are met, top the val set up to this "
                        "fraction of each hospital (0 to skip the top-up)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="_outputs/band")
    return p


def load_cache(path: str) -> dict:
    """Load the stats cache, refusing versions that lack per-patient counts."""
    with open(path) as f:
        stats = json.load(f)
    ver = stats.get("meta", {}).get("cache_version", 1)
    if ver < REQUIRED_CACHE_VERSION:
        raise SystemExit(
            f"{path} is cache_version {ver}; this needs >= "
            f"{REQUIRED_CACHE_VERSION} (per-patient occurrence counts). "
            f"Re-run hospital_stats.py without --from-cache to upgrade."
        )
    return stats


def choose_cohort(stats: dict, n_clients: int, min_records: int,
                  cohort_file: str = None) -> List[str]:
    """The cohort: an explicit manifest, else the largest eligible hospitals."""
    sizes = {h: r["n_records"] for h, r in stats["hospitals"].items()}
    if cohort_file:
        with open(cohort_file) as f:
            manifest = json.load(f)
        cohort = [str(h["hospital_id"]) for h in manifest["hospitals"]]
        missing = [h for h in cohort if h not in sizes]
        if missing:
            raise SystemExit(f"cohort hospitals absent from cache: {missing}")
        return cohort
    eligible = [h for h, n in sizes.items() if n >= min_records]
    if len(eligible) < n_clients:
        raise SystemExit(
            f"only {len(eligible)} hospitals have >= {min_records} records; "
            f"need {n_clients}. Lower --min-hospital-records.")
    return sorted(eligible, key=lambda h: -sizes[h])[:n_clients]


def select_cohort_patients(stats: dict, cohort: List[str], cap: int, seed: int
                           ) -> Dict[str, List[int]]:
    """The patients each hospital contributes to the study, capped if asked.

    Federated training cost scales with total data, so capping per hospital is
    the main compute dial.  Ordinals stay in the ORIGINAL dataset numbering (not
    re-indexed) so the manifest still points at real samples via
    hospital_stats.py's ``field_index``.

    Args:
        stats: The stats cache.
        cohort: Hospital ids.
        cap: Max patients per hospital, or None for all of them.
        seed: RNG seed; the sample is deterministic given it.

    Returns:
        ``{hospital_id: sorted patient ordinals in the study}``.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, List[int]] = {}
    for hid in cohort:
        n = stats["hospitals"][hid]["n_records"]
        if cap is None or cap >= n:
            out[hid] = list(range(n))
        else:
            out[hid] = sorted(rng.choice(n, size=cap, replace=False).tolist())
    return out


def restrict_stats(stats: dict, cohort: List[str],
                   cohort_patients: Dict[str, List[int]]) -> dict:
    """A view of the cache limited to the in-study patients.

    Every downstream count -- band selection, the split, per-pair totals -- must
    see only these patients, otherwise a capped run would still select codes on
    evidence it will not actually train on.
    """
    out = {"meta": stats["meta"], "hospitals": {}}
    for hid in cohort:
        rec = stats["hospitals"][hid]
        keep = set(cohort_patients[hid])
        ids: Dict[str, List[int]] = {}
        occ: Dict[str, List[int]] = {}
        for code, pids in rec["code_patient_ids"].items():
            pocc = rec["code_patient_occ"][code]
            sel = [(p, o) for p, o in zip(pids, pocc) if p in keep]
            if sel:
                ids[code] = [p for p, _ in sel]
                occ[code] = [o for _, o in sel]
        out["hospitals"][hid] = {
            "n_records": len(keep), "code_patient_ids": ids,
            "code_patient_occ": occ,
        }
    return out


def patient_counts(stats: dict, cohort: List[str]) -> Dict[str, Dict[str, int]]:
    """``{hospital: {code: n_patients}}`` restricted to the cohort."""
    return {h: {c: len(p)
                for c, p in stats["hospitals"][h]["code_patient_ids"].items()}
            for h in cohort}


def select_band(per_hosp: Dict[str, Dict[str, int]], val_quota: int,
                local_train_max: int, global_train_min: int
                ) -> List[Tuple[str, str]]:
    """The (hospital, code) pairs meeting all three band conditions.

    Args:
        per_hosp: ``{hospital: {code: n_patients}}`` over the cohort.
        val_quota: Held-out positives required at the hospital.
        local_train_max: Most patients that may remain at the hospital after the
            holdout for it to count as locally unlearnable.
        global_train_min: Cohort-wide training patients needed for the pair to be
            federated-learnable.

    Returns:
        Sorted ``[(hospital_id, code)]``.
    """
    global_pat: Dict[str, int] = {}
    for counts in per_hosp.values():
        for code, n in counts.items():
            global_pat[code] = global_pat.get(code, 0) + n

    # The global check here is optimistic: it subtracts a single quota, but a
    # code that is a band pair at several hospitals is held out at EACH of them
    # (and the val_frac top-up removes further carriers).  main() therefore
    # re-checks this condition against the realised split and drops any pair that
    # fails, iterating to a fixed point -- this is only the cheap first cut.
    band: List[Tuple[str, str]] = []
    for hid, counts in per_hosp.items():
        for code, n in counts.items():
            if n < val_quota:                                  # can't measure
                continue
            if n - val_quota > local_train_max:                # could learn alone
                continue
            if global_pat[code] - val_quota < global_train_min:  # nobody can learn
                continue
            band.append((hid, code))
    return sorted(band)


def cohort_train_patients(stats: dict, cohort: List[str], code: str,
                          val_sets: Dict[str, set]) -> int:
    """Patients carrying ``code`` left on the TRAIN side, cohort-wide."""
    total = 0
    for hid in cohort:
        ids = stats["hospitals"][hid]["code_patient_ids"].get(code)
        if not ids:
            continue
        total += sum(1 for p in ids if p not in val_sets[hid])
    return total


def resolve_band(stats: dict, cohort: List[str], band: List[Tuple[str, str]],
                 val_quota: int, global_train_min: int, val_frac: float,
                 seed: int, max_iters: int = 10,
                 cohort_patients: Dict[str, List[int]] = None):
    """Iterate band selection and splitting until every pair really qualifies.

    Selecting the band and building the split are mutually dependent: the split
    is sized by the band, and holding out quotas is what pushes a code's
    remaining train support below ``global_train_min``.  Dropping a pair only
    ever frees held-out patients, so repeatedly dropping violators converges
    (monotonically shrinking band).  It is mildly conservative -- a pair dropped
    early might have qualified once other pairs were dropped -- which errs
    toward a band whose members all genuinely satisfy the definition.

    Returns:
        ``(band, val_ids, n_quota, n_dropped)``.
    """
    dropped_total = 0
    for _ in range(max_iters):
        val_ids, n_quota = build_val_split(stats, cohort, band, val_quota,
                                           val_frac, seed, cohort_patients)
        val_sets = {h: set(v) for h, v in val_ids.items()}
        keep, dropped = [], 0
        for hid, code in band:
            if cohort_train_patients(stats, cohort, code,
                                     val_sets) >= global_train_min:
                keep.append((hid, code))
            else:
                dropped += 1
        if not dropped:
            return band, val_ids, n_quota, dropped_total
        dropped_total += dropped
        band = keep
        if not band:
            raise SystemExit(
                "Band emptied while enforcing --global-train-min. Lower it, "
                "lower --val-quota, or use more clients.")
    raise SystemExit(
        f"Band did not stabilise in {max_iters} iterations; loosen the "
        f"thresholds.")


def build_val_split(stats: dict, cohort: List[str],
                    band: List[Tuple[str, str]], val_quota: int,
                    val_frac: float, seed: int,
                    cohort_patients: Dict[str, List[int]] = None
                    ) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """Pick each hospital's held-out patients, quota-first then top-up.

    Codes are filled rarest-first (fewest carriers): the scarcest constraints are
    the hardest to satisfy, and patients drawn for them also count toward the
    commoner codes they happen to carry, so filling the other order strands the
    rare ones.  Within a code, carriers are drawn uniformly, keeping each code's
    held-out positives an unbiased sample of that hospital's carriers.

    Args:
        stats: The stats cache.
        cohort: Hospital ids.
        band: The (hospital, code) pairs to guarantee quota for.
        val_quota: Held-out patients required per pair.
        val_frac: Fraction of each hospital to reach after quotas (0 to skip).
        seed: RNG seed; the split is deterministic given it.

    Returns:
        ``(val_ids, n_quota_patients)`` -- held-out patient ordinals per
        hospital, and how many of them the quotas alone accounted for.
    """
    rng = np.random.default_rng(seed)
    by_hosp: Dict[str, List[str]] = {h: [] for h in cohort}
    for hid, code in band:
        by_hosp[hid].append(code)

    val_ids: Dict[str, List[int]] = {}
    n_quota: Dict[str, int] = {}
    for hid in cohort:
        rec = stats["hospitals"][hid]
        ids_by_code = rec["code_patient_ids"]
        chosen: set = set()
        # Rarest-first: fewest carriers at this hospital.
        for code in sorted(by_hosp[hid], key=lambda c: len(ids_by_code[c])):
            carriers = ids_by_code[code]
            have = sum(1 for p in carriers if p in chosen)
            need = val_quota - have
            if need <= 0:
                continue
            pool = [p for p in carriers if p not in chosen]
            take = rng.choice(len(pool), size=min(need, len(pool)),
                              replace=False)
            chosen.update(pool[int(i)] for i in take)
        n_quota[hid] = len(chosen)

        # Top up with uniformly random remaining patients so the held-out set is
        # a normal-sized evaluation slice, not just the quota patients (who are
        # band carriers and would skew it toward tail-heavy records).
        in_study = (cohort_patients[hid] if cohort_patients is not None
                    else list(range(rec["n_records"])))
        target = int(round(val_frac * len(in_study)))
        if target > len(chosen):
            rest = [p for p in in_study if p not in chosen]
            extra = rng.choice(len(rest), size=min(target - len(chosen),
                                                   len(rest)), replace=False)
            chosen.update(rest[int(i)] for i in extra)
        val_ids[hid] = sorted(chosen)
    return val_ids, n_quota


def split_counts(stats: dict, hid: str, code: str, val_set: set
                 ) -> Tuple[int, int, int, int]:
    """``(val_patients, val_occ, train_patients, train_occ)`` for one pair."""
    rec = stats["hospitals"][hid]
    ids = rec["code_patient_ids"][code]
    occ = rec["code_patient_occ"][code]
    v_pat = v_occ = t_pat = t_occ = 0
    for pid, n in zip(ids, occ):
        if pid in val_set:
            v_pat += 1
            v_occ += n
        else:
            t_pat += 1
            t_occ += n
    return v_pat, v_occ, t_pat, t_occ


def main():
    args = _build_arg_parser().parse_args()
    full_stats = load_cache(args.cache)
    cohort = choose_cohort(full_stats, args.n_clients, args.min_hospital_records,
                           args.cohort_file)
    full_sizes = {h: full_stats["hospitals"][h]["n_records"] for h in cohort}
    cohort_patients = select_cohort_patients(full_stats, cohort,
                                             args.cap_per_hospital, args.seed)
    # Everything downstream sees ONLY the in-study patients, so a capped run
    # never selects codes on evidence it will not train on.
    stats = restrict_stats(full_stats, cohort, cohort_patients)
    sizes = {h: len(cohort_patients[h]) for h in cohort}
    print(f"Cohort ({len(cohort)} hospitals, {sum(sizes.values())} patients"
          + (f", capped at {args.cap_per_hospital}/hospital from "
             f"{sum(full_sizes.values())}" if args.cap_per_hospital else "")
          + f"): {', '.join(f'{h}({sizes[h]})' for h in cohort)}")

    per_hosp = patient_counts(stats, cohort)
    band = select_band(per_hosp, args.val_quota, args.local_train_max,
                       args.global_train_min)
    if not band:
        raise SystemExit(
            "Band is empty. Loosen --local-train-max / --global-train-min, "
            "lower --val-quota, or use more clients.")
    print(f"Candidate band: {len(band)} (hospital, code) pairs over "
          f"{len({c for _, c in band})} codes")

    band, val_ids, n_quota, n_dropped = resolve_band(
        stats, cohort, band, args.val_quota, args.global_train_min,
        args.val_frac, args.seed, cohort_patients=cohort_patients)
    val_sets = {h: set(v) for h, v in val_ids.items()}
    codes = sorted({c for _, c in band})
    print(f"Final band:     {len(band)} pairs over {len(codes)} codes "
          f"({n_dropped} dropped: holding out the quota at every hospital that "
          f"has the code left < {args.global_train_min} to train on)")
    print(f"                quota {args.val_quota}, local_train <= "
          f"{args.local_train_max}, global_train >= {args.global_train_min}")

    # Per-pair counts on both sides of the split.
    rows = []
    for hid, code in band:
        v_pat, v_occ, t_pat, t_occ = split_counts(stats, hid, code,
                                                  val_sets[hid])
        g_train = sum(
            split_counts(stats, h2, code, val_sets[h2])[2]
            for h2 in cohort if code in stats["hospitals"][h2]["code_patient_ids"]
        )
        # Patient-weighted is the primary view: a patient counts once whether the
        # code appears in one visit or six. Prevalence (carriers / patients in
        # the slice) is the quantity a generator must reproduce and the positive
        # rate a per-code downstream metric is computed against. Occurrence
        # counts are kept as a secondary diagnostic -- they are dominated by a
        # few patients with many repeats and would otherwise skew every average.
        n_val_h = len(val_sets[hid])
        n_train_h = sizes[hid] - n_val_h
        rows.append({
            "hospital_id": hid, "icd9code": code,
            "val_patients": v_pat,
            "val_prevalence": round(v_pat / max(1, n_val_h), 6),
            "train_patients": t_pat,
            "train_prevalence": round(t_pat / max(1, n_train_h), 6),
            "hospital_total_patients": v_pat + t_pat,
            "cohort_train_patients": g_train,
            "val_occurrences": v_occ,
            "train_occurrences": t_occ,
            "val_occ_per_patient": round(v_occ / max(1, v_pat), 2),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "band_val_counts.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    manifest = {
        "cohort": cohort,
        "params": {
            "n_clients": len(cohort),
            "min_hospital_records": args.min_hospital_records,
            "cap_per_hospital": args.cap_per_hospital,
            "val_quota": args.val_quota,
            "local_train_max": args.local_train_max,
            "global_train_min": args.global_train_min,
            "val_frac": args.val_frac,
            "seed": args.seed,
        },
        "band_pairs": [{"hospital_id": h, "icd9code": c} for h, c in band],
        # The capped study population; training = these minus the val ordinals.
        "cohort_patient_ordinals": cohort_patients,
        "val_patient_ordinals": val_ids,
        "note": "Ordinals index each hospital's samples in dataset order, "
                "matching hospital_stats.py's field_index. Train on "
                "cohort_patient_ordinals minus val_patient_ordinals. Reuse this "
                "manifest byte-identically across all training regimes.",
    }
    man_path = os.path.join(args.out_dir, "tail_band_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- summary (patient-weighted first; occurrences are secondary) ------- #
    def _stat(vals):
        v = sorted(vals)
        return v[0], v[len(v) // 2], v[-1]

    print(f"\nPer-hospital val split and band coverage (patient-weighted):")
    print(f"  {'hospital':>9}  {'patients':>9}  {'val':>6}  {'val%':>6}  "
          f"{'quota pts':>10}  {'pairs':>6}  {'val carriers':>13}")
    print("  " + "-" * 72)
    for hid in cohort:
        h_rows = [r for r in rows if r["hospital_id"] == hid]
        n_val = len(val_ids[hid])
        print(f"  {hid:>9}  {sizes[hid]:>9}  {n_val:>6}  "
              f"{100.0 * n_val / sizes[hid]:>5.1f}%  {n_quota[hid]:>10}  "
              f"{len(h_rows):>6}  "
              f"{sum(r['val_patients'] for r in h_rows):>13}")
    tot_val = sum(len(v) for v in val_ids.values())
    print("  " + "-" * 72)
    print(f"  {'TOTAL':>9}  {sum(sizes.values()):>9}  {tot_val:>6}  "
          f"{100.0 * tot_val / sum(sizes.values()):>5.1f}%  "
          f"{sum(n_quota.values()):>10}  {len(rows):>6}  "
          f"{sum(r['val_patients'] for r in rows):>13}")

    lo, mid, hi = _stat(r["val_patients"] for r in rows)
    below = sum(1 for r in rows if r["val_patients"] < args.val_quota)
    print(f"\n  PRIMARY (patient-weighted)")
    print(f"    val carriers per pair  : min {lo}, median {mid}, max {hi}")
    print(f"    pairs below quota      : {below} of {len(rows)}"
          + ("" if not below else "  (fewer carriers than quota)"))
    lo, mid, hi = _stat(r["val_prevalence"] for r in rows)
    print(f"    val prevalence         : min {lo:.4f}, median {mid:.4f}, "
          f"max {hi:.4f}  (carriers / val patients)")
    lo, mid, hi = _stat(r["train_patients"] for r in rows)
    print(f"    local train carriers   : min {lo}, median {mid}, max {hi}  "
          f"(must stay <= {args.local_train_max})")
    lo, mid, hi = _stat(r["cohort_train_patients"] for r in rows)
    print(f"    cohort train carriers  : min {lo}, median {mid}, max {hi}  "
          f"(must stay >= {args.global_train_min})")

    lo, mid, hi = _stat(r["val_occ_per_patient"] for r in rows)
    print(f"\n  SECONDARY (occurrences -- diagnostic only)")
    print(f"    val occurrences        : {sum(r['val_occurrences'] for r in rows)}"
          f" total")
    print(f"    occurrences per carrier: min {lo:.2f}, median {mid:.2f}, "
          f"max {hi:.2f}")
    print(f"    -> repeats are concentrated in few patients; report "
          f"patient-weighted metrics as primary.")
    print(f"\nWrote:\n  {csv_path}\n  {man_path}")


if __name__ == "__main__":
    main()
