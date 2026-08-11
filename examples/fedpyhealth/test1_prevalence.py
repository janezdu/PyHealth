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
  the freeze step's < 5% rule), which is where a federated generator is meant
  to earn its keep.

The real reference is always the **held-out validation split**, never the
training split, so this measures out-of-sample fidelity rather than how well
the generator memorised what it saw.

Runs two ways, on one implementation:

* **in-run** -- ``train.py`` imports :func:`evaluate_rare_prevalence` and calls
  it while the generator's output and each hospital's val split are still in
  memory. Free.
* **standalone** -- this file's CLI re-scores a finished run from its saved
  ``synthetic.json``, so a metric fix costs a CPU job instead of repeating the
  GPU training run::

      python examples/fedpyhealth/main.py test1 \
          --cohort-file examples/fedpyhealth/cohorts/strat8.json \
          --run fedavg=_outputs/<run_name>_save

Results land in ``_outputs/results/tests/test1_prevalence.json``.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List

import pandas as pd

from utils.cohort_io import (
    EICU_ROOT,
    load_manifest,
    load_real_trajectories,
    load_synthetic,
    parse_run_specs,
    rare_codes_by_hospital,
    split_ids,
)
from pyhealth.metrics.generative.utility import compute_prevalence_metrics

DEFAULT_OUT = "_outputs/results/tests/test1_prevalence.json"

# The flat long-format schema every generative metric in PyHealth expects:
# one row per (patient, visit index, code).
EVAL_SCHEMA = {"visit_codes": str, "labels": int, "time": int, "id": str}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--cohort-file", required=True,
                   help="frozen manifest WITH patient ids (not *.summary.json)")
    p.add_argument("--run", action="append", default=[], metavar="NAME=SAVE_DIR",
                   help="a finished run to score; repeatable")
    p.add_argument("--eicu-root", default=EICU_ROOT)
    p.add_argument("--n-bootstraps", type=int, default=5,
                   help="bootstrap resamples over codes")
    p.add_argument("--dev", action="store_true",
                   help="load a small eICU subset (smoke only)")
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
    :func:`real_subset_to_records`: ``utils.cohort_io.load_real_trajectories`` has
    already decoded the index tensors into code strings.
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
    manifest: dict,
    per_hospital_synth: Dict[str, List[dict]],
    real: Dict[str, List[List[str]]],
    n_bootstraps: int = 5,
) -> Dict[str, dict]:
    """Standalone entry point: score every hospital of one finished run.

    Returns:
        ``{hospital_id: {metric: [mean, std]}}``.
    """
    val_by_hospital = split_ids(manifest, "val")
    rare_by_hospital = rare_codes_by_hospital(manifest)

    out: Dict[str, dict] = {}
    for hid, val_ids in val_by_hospital.items():
        synth = per_hospital_synth.get(hid)
        if not synth:
            print(f"  [{hid}] no synthetic patients in this run; skipped")
            continue
        val_traj = {p: real[p] for p in val_ids if p in real}
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

    manifest = load_manifest(args.cohort_file)
    val_ids = sorted({p for ids in split_ids(manifest, "val").values()
                      for p in ids})
    real = load_real_trajectories(args.eicu_root, set(val_ids), dev=args.dev)
    print(f"cohort: {len(manifest['hospitals'])} hospitals, "
          f"{len(val_ids)} validation patients")

    results = {}
    for name, save_dir in runs.items():
        print(f"\n=== {name}  ({save_dir})", flush=True)
        results[name] = score_run(
            manifest, load_synthetic(save_dir), real, args.n_bootstraps,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "kind": "test",
            "test": "test1_prevalence",
            "cohort_file": args.cohort_file,
            "cohort_name": manifest["meta"].get("cohort_name"),
            "n_bootstraps": args.n_bootstraps,
            "runs": results,
        }, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
