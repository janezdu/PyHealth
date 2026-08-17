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
    """Decode a real SampleDataset subset (index tensors) into long-format rows."""
    for sample in subset:
        pid = str(sample["patient_id"])
        for t, visit in enumerate(sample["visits"].tolist()):
            for idx in visit:
                code = index_to_code.get(int(idx))
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
) -> Dict[str, dict]:
    """Standalone entry point: score every hospital of one finished run.

    Args:
        per_hospital_synth: ``{hospital_id: [synthetic patient, ...]}``.
        real: ``{hospital_id: {patient_id: [[code, ...], ...]}}`` for the
            scoring fold, straight from the cohort cache.
        rare_by_hospital: Each hospital's rare codes.
        n_bootstraps: Bootstrap resamples over codes.
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
        val_df = pd.DataFrame(trajectories_to_records(val_traj)).astype(EVAL_SCHEMA)
        syn_df = pd.DataFrame(synthetic_to_records(synth)).astype(EVAL_SCHEMA)
        scores = prevalence_from_frames(
            val_df, syn_df, rare_by_hospital.get(hid), n_bootstraps, label=hid,
        )
        r2 = scores.get("PrevVal_All_Prevalence_R2", (float("nan"),))[0]
        rare_r2 = scores.get("PrevVal_Rare_Prevalence_R2", (float("nan"),))[0]
        print(f"  [{hid}] {len(val_traj)} val patients, {len(synth)} synthetic"
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
    n_real = sum(len(v) for v in real.values())
    print(f"cohort '{manifest.get('cohort_name')}': {len(real)} hospitals, "
          f"{n_real} {args.fold} patients")

    results = {}
    for name, save_dir in runs.items():
        print(f"\n=== {name}  ({save_dir})", flush=True)
        results[name] = score_run(
            load_synthetic(save_dir), real, rare_by_hospital, args.n_bootstraps,
            synth_cap=args.synth_cap,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "kind": "test",
            "test": "test1_prevalence",
            "cohort_cache": args.cohort_cache,
            "cohort_name": manifest.get("cohort_name"),
            "fold": args.fold,
            "n_bootstraps": args.n_bootstraps,
            "synth_cap": args.synth_cap,
            "runs": results,
        }, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
