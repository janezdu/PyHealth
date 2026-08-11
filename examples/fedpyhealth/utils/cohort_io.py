"""Shared loaders for a frozen cohort and a finished run's artifacts.

Both tests need the same three things -- the frozen manifest, the cohort's real
patient trajectories, and a run's persisted synthetic patients -- so they live
here rather than in either test, which would make one test import the other.

Nothing in this module trains or scores anything; it only reads.
"""

import json
import os
from typing import Dict, List, Sequence, Set

# Kept in sync with train.py so every stage sees the same task samples.
EICU_ROOT = "/work/hdd/bgyw/janezdu/data/eicu/eicu-crd/2.0"
MIN_VISITS = 1
DEFAULT_COHORT_FILE = "examples/fedpyhealth/cohorts/strat8.json"


def load_manifest(path: str) -> dict:
    """Read a frozen cohort manifest, rejecting one with no frozen split.

    Checks for the fields it actually needs rather than a version number: a
    selection file (``*.cohort.json``) names the hospitals but assigns no
    patients, and hitting that difference as a ``KeyError`` an hour into a GPU
    job is a bad way to find out.

    Args:
        path: Path to the manifest written by the freeze step (the one WITH
            patient ids, not the ``*.summary.json`` aggregate).

    Returns:
        The manifest dict.

    Raises:
        FileNotFoundError: If the manifest is missing -- with a pointer to the
            likely cause, which is that the cohort was selected but never frozen.
        ValueError: If the file carries no per-hospital train/val patient ids.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. A *.cohort.json file lists hospital ids only; "
            "the frozen split (train/val patient ids per hospital) comes from "
            "the freeze step. Run it before training or scoring."
        )
    with open(path) as fh:
        manifest = json.load(fh)

    hospitals = manifest.get("hospitals")
    if not hospitals:
        raise ValueError(f"{path} has no 'hospitals' block")
    missing = [k for k in ("train_patient_ids", "val_patient_ids")
               if k not in hospitals[0]]
    if missing:
        raise ValueError(
            f"{path} carries no frozen split (missing {', '.join(missing)}). "
            "It is a selection file, not a freeze manifest -- run "
            "prepare_dataset.py freeze on it first, so every regime trains and "
            "scores on byte-identical data."
        )
    return manifest


def hospital_ids(manifest: dict) -> List[str]:
    """Cohort hospital ids, in manifest order."""
    return [h["hospital_id"] for h in manifest["hospitals"]]


def split_ids(manifest: dict, fold: str) -> Dict[str, List[str]]:
    """Map hospital id -> that hospital's frozen ``train``/``val`` patient ids."""
    key = f"{fold}_patient_ids"
    return {h["hospital_id"]: list(h[key]) for h in manifest["hospitals"]}


def rare_codes_by_hospital(manifest: dict) -> Dict[str, List[str]]:
    """Map hospital id -> its own rare codes (rare *at that hospital*)."""
    return {h["hospital_id"]: sorted(h.get("rare_codes", {}))
            for h in manifest["hospitals"]}


def load_real_trajectories(
    eicu_root: str, wanted: Set[str], dev: bool = False
) -> Dict[str, List[List[str]]]:
    """Decode the cohort's real patients back to code-string trajectories.

    Raises:
        ValueError: If a manifest patient is absent from this eICU build, which
            means the manifest is stale relative to the data.
    """
    from pyhealth.datasets import eICUDataset
    from pyhealth.tasks import EHRGenerationEICU

    print(f"Loading eICU from {eicu_root} (dev={dev})...", flush=True)
    base = eICUDataset(root=eicu_root, tables=["diagnosis"], dev=dev)
    samples = base.set_task(EHRGenerationEICU(min_visits=MIN_VISITS))
    index_to_code = {
        v: k for k, v in samples.input_processors["visits"].code_vocab.items()
    }
    out: Dict[str, List[List[str]]] = {}
    for i in range(len(samples)):
        sample = samples[i]
        pid = str(sample["patient_id"])
        if pid not in wanted:
            continue
        visits = []
        for visit in sample["visits"].tolist():
            codes = [index_to_code.get(int(c)) for c in visit]
            codes = [c for c in codes if c not in (None, "<pad>", "<unk>")]
            if codes:
                visits.append(codes)
        if visits:
            out[pid] = visits
    missing = wanted - set(out)
    if missing:
        raise ValueError(
            f"{len(missing)} manifest patients absent from this eICU build, "
            f"e.g. {sorted(missing)[:5]}. The manifest is stale -- re-freeze it."
        )
    return out


def load_synthetic(save_dir: str) -> Dict[str, List[dict]]:
    """Read a run's persisted per-hospital synthetic patients.

    Raises:
        FileNotFoundError: If the run has no ``synthetic.json`` (a job that died
            before the final save step).
        ValueError: If the file predates the per-hospital protocol.
    """
    path = os.path.join(save_dir, "synthetic.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is written at the very end of train.py; a "
            "run that timed out first has no synthetic data to score."
        )
    with open(path) as fh:
        blob = json.load(fh)
    if "per_hospital" not in blob:
        raise ValueError(
            f"{path} has no 'per_hospital' block -- it predates the "
            "per-hospital protocol. Re-run the regime."
        )
    return blob["per_hospital"]


def parse_run_specs(specs: Sequence[str]) -> Dict[str, str]:
    """Turn ``["fedavg=_outputs/x_save", ...]`` into ``{name: save_dir}``.

    Raises:
        ValueError: If a spec is not ``NAME=SAVE_DIR``.
    """
    runs: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--run {spec!r} must look like NAME=SAVE_DIR")
        name, save_dir = spec.split("=", 1)
        runs[name.strip()] = save_dir.strip()
    return runs
