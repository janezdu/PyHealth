"""Test 1: prevalence fidelity -- does synthetic data keep each code's rate?

The question this answers is *overall accuracy*, not the long tail: a code
carried by 20% of real patients should be carried by roughly 20% of generated
ones. Per code, prevalence is the fraction of patients carrying it at least
once; the real and synthetic prevalence vectors are then compared by R^2,
Pearson and RMSE, bootstrapped over codes. Test 2
(``test2_rare_efficacy.py``) is the long-tail counterpart.

Two variants are reported per hospital:

* ``PrevVal_All_*``  -- the full code vocabulary.
* ``PrevVal_Rare_*`` -- only that hospital's own rare codes (rare *there*, per
  the cache's <= 5% rule), which is where a federated generator is meant to
  earn its keep.

The real reference is always a **held-out fold**, never the training split, so
this measures out-of-sample fidelity rather than memorisation. It defaults to
validation; test stays untouched until the final numbers.

Runs two ways, on one implementation:

* **in-run** -- ``train.py`` imports :func:`evaluate_rare_prevalence` and calls
  it while the generator's output and each hospital's val split are still in
  memory. Free.
* **standalone** -- this file's CLI re-scores a finished run from its saved
  ``synthetic.json``, so a metric fix costs a CPU job instead of repeating the
  GPU training run::

      python examples/fedpyhealth/main.py test1 \
          --run fedavg=_outputs/<run_name>_save

Results land in ``_outputs/results/tests/test1_prevalence.json``.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List

import pandas as pd

from utils.cohort import (
    DEFAULT_CACHE_DIR,
    load_manifest,
    load_synthetic,
    parse_run_specs,
    rare_codes,
    read_trajectories,
)
from pyhealth.metrics.generative.utility import compute_prevalence_metrics

DEFAULT_OUT = "_outputs/results/tests/test1_prevalence.json"

# The flat long-format schema every generative metric in PyHealth expects:
# one row per (patient, visit index, code).
EVAL_SCHEMA = {"visit_codes": str, "labels": int, "time": int, "id": str}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR,
                   help="cohort cache directory built by utils/cohort.py")
    p.add_argument("--run", action="append", default=[], metavar="NAME=SAVE_DIR",
                   help="a finished run to score; repeatable")
    p.add_argument("--fold", default="val", choices=["val", "test"],
                   help="real fold to score against (default: val -- keep test "
                        "held out until the final numbers)")
    p.add_argument("--n-bootstraps", type=int, default=5,
                   help="bootstrap resamples over codes")
    p.add_argument("--real-scope", choices=["hospital", "pooled"],
                   default="hospital",
                   help="what each hospital's synthetic set is scored AGAINST. "
                        "'hospital' (default) compares synth[h] with that "
                        "site's own test fold -- the natural question, but the "
                        "real side is tiny at small sites (40 test patients at "
                        "429, so prevalence resolves only to 1/40) and every "
                        "site is measured against a different target. 'pooled' "
                        "compares every site's synthetic set with the SAME "
                        "cohort-wide test fold (2,430 patients, resolution "
                        "1/2430), which asks a different question: how far is "
                        "this site's generator from the cohort distribution. "
                        "Note that fedavg and centralized share one generator, "
                        "so under 'pooled' all eight of their per-hospital rows "
                        "are the identical comparison and will be identical.")
    p.add_argument("--rare-scope", choices=["hospital", "pooled"],
                   default="hospital",
                   help="which codes PrevVal_Rare_* scores. 'hospital' "
                        "(default) uses each site's own rare set -- <=5% "
                        "prevalence THERE -- so the tail is a different set of "
                        "codes at every site (296 at 458, 103 at 429) and the "
                        "per-hospital numbers are not strictly comparable. "
                        "'pooled' uses one shared definition, the union of "
                        "every site's rare set (548 codes, rare at >=1 "
                        "hospital), intersected with the codes that site "
                        "actually has in the scoring fold. The intersection is "
                        "not optional: 79% of the pooled pool has zero "
                        "prevalence at hospital 429 and 83% at 358, and scoring "
                        "those would add hundreds of (real 0, synth ~0) points "
                        "that agree trivially and inflate Pearson and R^2 "
                        "while measuring nothing.")
    p.add_argument("--synth-cap", type=int, default=0,
                   help="score only the first N synthetic patients per "
                        "hospital (0 = all). Prevalence resolves only to "
                        "1/N, so an arm holding more synthetic patients "
                        "scores a better R^2 for reasons unrelated to its "
                        "generator; cap every arm at the smallest to compare "
                        "fidelity rather than sample size")
    p.add_argument("--out", default=DEFAULT_OUT)
    return p


# --------------------------------------------------------------------------- #
# Record building                                                              #
# --------------------------------------------------------------------------- #
def real_subset_to_records(subset, index_to_code: Dict[int, str]):
    """Decode a real SampleDataset subset (multi-hot rows) into long-format rows.

    ``visits`` is ``(n_visits, vocab_size)`` and a code's identity is its COLUMN,
    not the stored value -- every stored value is 0.0 or 1.0. Reading the values
    as vocabulary indices (which is what the index-tensor form required) decodes
    every visit to ``<pad>``/``<unk>``, yields nothing, and leaves the caller
    with an empty frame rather than an error.
    """
    for sample in subset:
        pid = str(sample["patient_id"])
        for t, visit in enumerate(sample["visits"]):
            for col in visit.nonzero().flatten().tolist():
                code = index_to_code.get(int(col))
                if code in (None, "<pad>", "<unk>"):
                    continue
                yield {"id": pid, "time": t, "visit_codes": code, "labels": 0}


def synthetic_to_records(patients: List[Dict]):
    """Convert generator output [{patient_id, visits:[[code]]}] into long rows."""
    for p in patients:
        pid = str(p["patient_id"])
        for t, visit in enumerate(p["visits"]):
            for code in visit:
                yield {"id": pid, "time": t, "visit_codes": str(code), "labels": 0}


def trajectories_to_records(trajectories: Dict[str, List[List[str]]]):
    """Convert ``{patient_id: [[code, ...], ...]}`` into long-format rows.

    This is the standalone path's equivalent of
    :func:`real_subset_to_records`: ``utils.cohort.read_trajectories`` reads the
    cache's code strings directly, so there are no index tensors to decode.
    """
    for pid, visits in trajectories.items():
        for t, visit in enumerate(visits):
            for code in visit:
                yield {"id": str(pid), "time": t, "visit_codes": str(code),
                       "labels": 0}


def _prefixed(results: Dict[str, tuple], prefix: str) -> Dict[str, tuple]:
    """Namespace a metric dict so All/Rare variants never collide."""
    return {f"{prefix}{k}": v for k, v in results.items()}


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def prevalence_from_frames(
    val_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    rare_codes: Iterable[str] = None,
    n_bootstraps: int = 5,
    label: str = "global",
) -> Dict[str, tuple]:
    """Score one real/synthetic frame pair, All and (optionally) Rare.

    ``code_subset`` is used rather than pre-filtering rows: filtering would also
    shrink the per-patient denominator, and by a different factor in each frame,
    which biases R^2 and RMSE (see ``compute_prevalence_metrics``).

    Returns:
        ``{metric_name: (mean, std)}``, empty if either frame is empty.
    """
    if val_df.empty or syn_df.empty:
        print(f"  [{label}] prevalence skipped: empty frame")
        return {}

    out: Dict[str, tuple] = {}
    out.update(_prefixed(
        compute_prevalence_metrics(val_df, syn_df, n_bootstraps=n_bootstraps),
        "PrevVal_All_",
    ))
    if rare_codes:
        out.update(_prefixed(
            compute_prevalence_metrics(
                val_df, syn_df, n_bootstraps=n_bootstraps,
                code_subset=list(rare_codes),
            ),
            "PrevVal_Rare_",
        ))
    else:
        print(f"  [{label}] no rare codes supplied; PrevVal_Rare_* skipped")
    return out


def evaluate_rare_prevalence(
    val_subset,
    synthetic,
    index_to_code: Dict[int, str],
    rare_codes: List[str] = None,
    n_bootstraps: int = 5,
    label: str = "global",
) -> Dict[str, tuple]:
    """In-run entry point: score a hospital straight from live objects.

    Args:
        val_subset: The hospital's held-out real samples (SampleDataset subset).
        synthetic: Generated patients, ``[{"patient_id", "visits"}, ...]``.
        index_to_code: Inverted code vocabulary for decoding the real samples.
        rare_codes: This hospital's rare codes. Falsy skips the rare variant.
        n_bootstraps: Bootstrap resamples over codes.
        label: Tag used in log lines.

    Returns:
        ``{metric_name: (mean, std)}``.
    """
    val_df = pd.DataFrame(
        real_subset_to_records(val_subset, index_to_code)
    ).astype(EVAL_SCHEMA)
    syn_df = pd.DataFrame(synthetic_to_records(synthetic)).astype(EVAL_SCHEMA)
    return prevalence_from_frames(val_df, syn_df, rare_codes, n_bootstraps, label)


def score_run(
    per_hospital_synth: Dict[str, List[dict]],
    real: Dict[str, Dict[str, List[List[str]]]],
    rare_by_hospital: Dict[str, List[str]],
    n_bootstraps: int = 5,
    synth_cap: int = 0,
    pooled_real: "pd.DataFrame" = None,
) -> Dict[str, dict]:
    """Standalone entry point: score every hospital of one finished run.

    Args:
        per_hospital_synth: ``{hospital_id: [synthetic patient, ...]}``.
        real: ``{hospital_id: {patient_id: [[code, ...], ...]}}`` for the
            scoring fold, straight from the cohort cache.
        rare_by_hospital: Each hospital's rare codes.
        n_bootstraps: Bootstrap resamples over codes.
        pooled_real: When given, every hospital's synthetic set is scored
            against THIS frame instead of its own test fold. Changes the
            question from "does site h's generator match site h" to "does it
            match the cohort", and lifts the real side from as few as 40
            patients to 2,430 -- which matters because prevalence resolves only
            to 1/n_real, so a 40-patient fold cannot express anything below
            0.025 no matter how much synthetic data it is compared with.
        synth_cap: Score only the first N synthetic patients per hospital;
            0 uses all of them. A synthetic set of size N can only express
            prevalences in multiples of 1/N, so an arm with more synthetic
            patients gets a less quantized estimate and a better R^2 for
            reasons that have nothing to do with its generator. Capping every
            arm at the smallest one makes the comparison about fidelity again.

    Returns:
        ``{hospital_id: {metric: [mean, std]}}``.
    """
    out: Dict[str, dict] = {}
    for hid, val_traj in real.items():
        synth = per_hospital_synth.get(hid)
        if not synth:
            print(f"  [{hid}] no synthetic patients in this run; skipped")
            continue
        if synth_cap > 0:
            if len(synth) < synth_cap:
                raise ValueError(
                    f"hospital {hid} has {len(synth)} synthetic patients but "
                    f"--synth-cap is {synth_cap}; lower the cap so every arm "
                    "can meet it, or the comparison is not matched after all."
                )
            synth = synth[:synth_cap]
        val_df = (pooled_real if pooled_real is not None
                  else pd.DataFrame(
                      trajectories_to_records(val_traj)).astype(EVAL_SCHEMA))
        syn_df = pd.DataFrame(synthetic_to_records(synth)).astype(EVAL_SCHEMA)
        scores = prevalence_from_frames(
            val_df, syn_df, rare_by_hospital.get(hid), n_bootstraps, label=hid,
        )
        r2 = scores.get("PrevVal_All_Prevalence_R2", (float("nan"),))[0]
        rare_r2 = scores.get("PrevVal_Rare_Prevalence_R2", (float("nan"),))[0]
        n_real = (val_df["id"].nunique() if pooled_real is not None
                  else len(val_traj))
        print(f"  [{hid}] {n_real} val patients, {len(synth)} synthetic"
              f"  ->  all R2={r2:.4f}  rare R2={rare_r2:.4f}", flush=True)
        out[hid] = {k: list(v) for k, v in scores.items()}
    return out


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    runs = parse_run_specs(args.run)
    if not runs:
        raise SystemExit("nothing to score: pass at least one --run NAME=SAVE_DIR")

    manifest = load_manifest(args.cohort_cache)
    real = read_trajectories(args.cohort_cache, args.fold)
    rare_by_hospital = rare_codes(args.cohort_cache)
    pooled_real = None
    if args.real_scope == "pooled":
        pooled_traj = {f"{hid}::{pid}": tr
                       for hid, pats in real.items() for pid, tr in pats.items()}
        pooled_real = pd.DataFrame(
            trajectories_to_records(pooled_traj)).astype(EVAL_SCHEMA)
        print(f"real scope: POOLED -- every arm's per-hospital synthetic set is "
              f"scored against the\n  same {len(pooled_traj)}-patient "
              f"cohort-wide {args.fold} fold (resolution 1/{len(pooled_traj)})")

    if args.rare_scope == "pooled":
        pooled = set(manifest["pooled_rare_codes"])
        scoped = {}
        if args.real_scope == "pooled":
            # One target frame means one code set: intersect with what the
            # POOLED fold holds, not with each site's own, or a hospital would
            # be scored on codes its comparison target never contains.
            present = set(pooled_real["visit_codes"].unique())
            shared = sorted(pooled & present)
            scoped = {hid: shared for hid in real}
        else:
            for hid, pats in real.items():
                present = {c for tr in pats.values() for v in tr for c in v}
                scoped[hid] = sorted(pooled & present)
        target = ("the POOLED {} fold -- one identical code set for every "
                  "hospital".format(args.fold) if args.real_scope == "pooled"
                  else "each site's own {} fold -- so the set still differs "
                       "per site".format(args.fold))
        print(f"rare scope: POOLED -- one shared definition of rare "
              f"({len(pooled)} codes, rare at >=1 hospital),\n  intersected "
              f"with {target}:")
        for hid in real:
            print(f"  {hid}: {len(scoped[hid])} of {len(pooled)} present "
                  f"(own rare set is {len(rare_by_hospital.get(hid, []))})")
        rare_by_hospital = scoped
    else:
        print("rare scope: PER-HOSPITAL -- each site's own <=5% codes; the tail "
              "is a different\n  set of codes at every site, so read "
              "PrevVal_Rare_* within a hospital, not across")
    n_real = sum(len(v) for v in real.values())
    print(f"cohort '{manifest.get('cohort_name')}': {len(real)} hospitals, "
          f"{n_real} {args.fold} patients")

    results = {}
    for name, save_dir in runs.items():
        print(f"\n=== {name}  ({save_dir})", flush=True)
        results[name] = score_run(
            load_synthetic(save_dir), real, rare_by_hospital, args.n_bootstraps,
            synth_cap=args.synth_cap, pooled_real=pooled_real,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "kind": "test",
            "test": "test1_prevalence",
            "cohort_cache": args.cohort_cache,
            "cohort_name": manifest.get("cohort_name"),
            "fold": args.fold,
            "rare_scope": args.rare_scope,
            "real_scope": args.real_scope,
            "n_bootstraps": args.n_bootstraps,
            "synth_cap": args.synth_cap,
            "runs": results,
        }, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
