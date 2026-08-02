"""Test 2: machine-learning efficacy on pooled rare codes (TSTR).

The question: if you train a downstream model on a regime's *synthetic* EHR,
how well does it predict which rare codes a real patient carries? This is the
standard Train-on-Synthetic, Test-on-Real (TSTR) protocol, specialised to the
tail of the code distribution.

The task
--------
Labels are multi-hot over ``pooled_rare_codes`` -- the union of all 8 hospitals'
rare-code sets, frozen in the cohort manifest. Hospital membership is
deliberately ignored when labelling: a synthetic patient has no hospital, so a
per-hospital label space would be undefined for exactly the data we need to
score. The pooled vocabulary is the only definition that applies to real and
synthetic patients alike.

Inputs are the patient's trajectory with **every** pooled rare code stripped
out. Masking only a patient's own positives would leave a "nothing was removed
here" asymmetry that the model can read straight off, so the mask is applied
uniformly to real train, real val and every synthetic set.

What is compared
----------------
=================  =====================================  ====================
model              training data                          role
=================  =====================================  ====================
prior              none (per-code training prevalence)     floor
trtr               masked real pooled train                ceiling
tstr:<regime>      masked synthetic from that regime       the comparison
=================  =====================================  ====================

Scored on the **same** real pooled validation set with the **same** frozen
processors, so the only thing that varies is the training data.

Reading the numbers
-------------------
Two failure modes look like a score but are not one, so both are instrumented:

1. If a generator emits almost no rare codes, its TSTR model sees near-all-zero
   targets and collapses to the prior. That is a real and important finding, but
   it is indistinguishable from a plumbing bug unless you look at the
   ``diagnostics`` block -- so a run with zero distinct rare codes in its
   synthetic data is reported as ``degenerate: true`` rather than as a score.
2. ``pooled_rare_codes`` mixes genuinely rare codes with codes that are rare at
   one hospital and common at another. A macro average over all of them is
   separated mostly by the easy head, so the headline is restricted to columns
   with at least ``--min-positives`` positives in the validation set.

Runs on one GPU. Loads eICU once and scores every regime, so submit it after the
training jobs finish -- it consumes the ``synthetic.json`` each run persists.

Example:
    python examples/fedpyhealth/rare_code_efficacy.py \\
        --cohort-file examples/fedpyhealth/cohorts/rare8_v1.json \\
        --run centralized=_outputs/centralized_E2_R20_rare8_v1_utility_save \\
        --run local=_outputs/local_E2_R20_rare8_v1_utility_save
"""

import argparse
import json
import os
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.processors import MultiLabelProcessor, NestedSequenceProcessor
from pyhealth.trainer import Trainer

EICU_ROOT = "/work/hdd/bgyw/janezdu/data/eicu/eicu-crd/2.0"
MIN_VISITS = 1
SEED = 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-file",
                   default="examples/fedpyhealth/cohorts/rare8_v1.json",
                   help="frozen manifest (schema_version >= 2)")
    p.add_argument("--run", action="append", default=[], metavar="NAME=SAVE_DIR",
                   help="a regime to score; SAVE_DIR is the run's _outputs/"
                        "<run_name>_save/ folder holding synthetic.json. "
                        "Repeatable.")
    p.add_argument("--eicu-root", default=EICU_ROOT)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--min-positives", type=int, default=5,
                   help="a rare code needs this many positives in the pooled "
                        "validation set to be scored (average_precision is "
                        "undefined at 0 and meaningless at 1)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recall-at", default="5,10,20",
                   help="comma-separated k values for recall@k")
    p.add_argument("--out", default="_outputs/results/test2_rare_efficacy.json")
    return p


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #
def mask_and_label(
    visits: Sequence[Sequence[str]], rare_pool: Set[str]
) -> Tuple[List[List[str]], List[str]]:
    """Strip every pooled rare code from the input; keep them as the label.

    Args:
        visits: The patient's trajectory as lists of code strings.
        rare_pool: Every pooled rare code (not just this patient's).

    Returns:
        ``(masked_visits, labels)``. Emptied visits are dropped; a patient whose
        codes are *all* rare ends up with no visits and must be discarded by the
        caller (there is nothing left to predict from).
    """
    masked, labels = [], set()
    for visit in visits:
        kept = []
        for code in visit:
            if code in rare_pool:
                labels.add(code)
            else:
                kept.append(code)
        if kept:
            masked.append(kept)
    return masked, sorted(labels)


def build_records(
    trajectories: Dict[str, List[List[str]]], rare_pool: Set[str], tag: str
) -> Tuple[List[dict], dict]:
    """Turn ``{patient_id: visits}`` into masked multilabel samples.

    Returns:
        ``(records, diagnostics)``. Diagnostics report how much of the cohort
        survived masking and how much rare-code signal is present -- the numbers
        that tell a real result apart from a degenerate one.
    """
    records, dropped = [], 0
    n_with_label, label_counts, distinct = 0, [], set()
    for pid, visits in trajectories.items():
        masked, labels = mask_and_label(visits, rare_pool)
        if not masked:
            dropped += 1
            continue
        records.append({
            "patient_id": str(pid),
            "visits": masked,
            "rare_labels": labels,
        })
        if labels:
            n_with_label += 1
        label_counts.append(len(labels))
        distinct.update(labels)

    n = max(1, len(records))
    diagnostics = {
        "n_records": len(records),
        "n_dropped_empty_input": dropped,
        "frac_with_ge1_rare_label": round(n_with_label / n, 4),
        "mean_rare_labels_per_record": round(float(np.mean(label_counts or [0])), 4),
        "n_distinct_rare_codes": len(distinct),
    }
    print(f"  [{tag}] {diagnostics}", flush=True)
    return records, diagnostics


def load_real_trajectories(
    eicu_root: str, wanted: Set[str], dev: bool = False
) -> Dict[str, List[List[str]]]:
    """Decode the cohort's real patients back to code-string trajectories."""
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
            f"e.g. {sorted(missing)[:5]}. The manifest is stale -- re-run "
            "freeze_cohort_split.py."
        )
    return out


def load_synthetic(save_dir: str) -> Dict[str, List[List[str]]]:
    """Read a run's persisted pooled synthetic patients."""
    path = os.path.join(save_dir, "synthetic.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is written at the end of ehr_eicu.py; a run "
            "that timed out before STEP 9 has no synthetic data to score."
        )
    with open(path) as fh:
        blob = json.load(fh)
    return {
        str(p["patient_id"]): [list(v) for v in p["visits"]]
        for p in blob["pooled"]
    }


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def make_dataset(records, input_proc, label_proc):
    """Build a SampleDataset that reuses already-fitted processors.

    Passing pre-fitted processors makes ``SampleBuilder.fit`` skip fitting, so
    the code vocabulary and label space are identical across the real train set,
    the real validation set and every regime's synthetic set. Without that, each
    TSTR model would be scored against a different label axis.
    """
    return create_sample_dataset(
        samples=records,
        input_schema={"visits": "nested_sequence"},
        output_schema={"rare_labels": "multilabel"},
        input_processors={"visits": input_proc},
        output_processors={"rare_labels": label_proc},
        dataset_name="rare_code_efficacy",
        task_name="rare_code_multilabel",
        in_memory=True,
    )


def train_and_score(
    train_records, val_dataset, input_proc, label_proc, args, tag: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Train one RNN on ``train_records`` and return ``(y_true, y_prob)``."""
    torch.manual_seed(SEED)
    train_dataset = make_dataset(train_records, input_proc, label_proc)
    model = RNN(
        dataset=train_dataset,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    # enable_logging=False keeps Trainer from creating ./output/<timestamp>/
    # checkpoint dirs in the repo root; we only need the fitted weights in RAM.
    trainer = Trainer(model=model, metrics=["pr_auc_macro"],
                      enable_logging=False)
    trainer.train(
        train_dataloader=get_dataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        ),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor=None,
        load_best_model_at_last=False,
    )
    y_true, y_prob, _ = trainer.inference(
        get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    )
    print(f"  [{tag}] trained on {len(train_records)} records", flush=True)
    return np.asarray(y_true), np.asarray(y_prob)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Patient-averaged fraction of true rare codes inside the top-k predicted.

    Only patients with at least one positive contribute; the metric is
    undefined for the rest.
    """
    scores = []
    order = np.argsort(-y_prob, axis=1)[:, :k]
    for row, top in zip(y_true, order):
        n_pos = int(row.sum())
        if n_pos == 0:
            continue
        scores.append(float(row[top].sum()) / min(n_pos, k))
    return float(np.mean(scores)) if scores else float("nan")


def score(
    y_true: np.ndarray, y_prob: np.ndarray, keep: np.ndarray, ks: Sequence[int]
) -> dict:
    """Macro AUPRC / AUROC over scorable columns, plus recall@k over all."""
    from sklearn import metrics as skm

    sub_true, sub_prob = y_true[:, keep], y_prob[:, keep]
    out = {
        "pr_auc_macro": float(skm.average_precision_score(
            sub_true, sub_prob, average="macro")),
        "pr_auc_micro": float(skm.average_precision_score(
            sub_true, sub_prob, average="micro")),
        "n_labels_scored": int(keep.sum()),
    }
    try:
        out["roc_auc_macro"] = float(skm.roc_auc_score(
            sub_true, sub_prob, average="macro"))
    except ValueError:
        out["roc_auc_macro"] = float("nan")  # a column with one class only
    for k in ks:
        out[f"recall_at_{k}"] = recall_at_k(y_true, y_prob, k)
    return out


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    ks = [int(k) for k in args.recall_at.split(",") if k.strip()]

    with open(args.cohort_file) as fh:
        manifest = json.load(fh)
    if int(manifest.get("schema_version", 1)) < 2:
        raise ValueError(
            f"{args.cohort_file} is a legacy manifest with no frozen splits or "
            "pooled rare codes; Test 2 needs a schema_version >= 2 manifest."
        )
    rare_pool = sorted(manifest["meta"]["pooled_rare_codes"])
    rare_set = set(rare_pool)
    train_ids, val_ids = set(), set()
    for h in manifest["hospitals"]:
        train_ids.update(h["train_patient_ids"])
        val_ids.update(h["val_patient_ids"])
    print(f"pooled rare codes: {len(rare_pool)}   "
          f"real train: {len(train_ids)}   real val: {len(val_ids)}")

    real = load_real_trajectories(
        args.eicu_root, train_ids | val_ids, dev=args.dev
    )
    print("\nBuilding masked records:")
    train_records, train_diag = build_records(
        {p: real[p] for p in sorted(train_ids)}, rare_set, "real-train")
    val_records, val_diag = build_records(
        {p: real[p] for p in sorted(val_ids)}, rare_set, "real-val")

    # Freeze the feature space on the real training data. Codes a generator
    # invents that were never seen in real training map to <unk>, which is the
    # correct TSTR behaviour.
    input_proc = NestedSequenceProcessor()
    input_proc.fit(train_records, "visits")
    label_proc = MultiLabelProcessor()
    label_proc.fit([{"rare_labels": rare_pool}], "rare_labels")
    print(f"frozen vocab: {input_proc.vocab_size()} codes, "
          f"{label_proc.size()} labels")

    val_dataset = make_dataset(val_records, input_proc, label_proc)

    # Only score columns with enough validation positives -- average_precision
    # is undefined at 0 positives and statistically empty at 1.
    val_pos = np.zeros(len(rare_pool))
    for rec in val_records:
        for code in rec["rare_labels"]:
            val_pos[label_proc.label_vocab[code]] += 1
    keep = val_pos >= args.min_positives
    print(f"scorable labels (>= {args.min_positives} val positives): "
          f"{int(keep.sum())}/{len(rare_pool)}")
    if not keep.any():
        raise ValueError(
            "no rare code has enough validation positives to score; lower "
            "--min-positives or widen --rare-prevalence-max in the freeze step"
        )

    results = {
        "cohort_file": args.cohort_file,
        "n_pooled_rare_codes": len(rare_pool),
        "min_positives": args.min_positives,
        "n_labels_scored": int(keep.sum()),
        "real_train_diagnostics": train_diag,
        "real_val_diagnostics": val_diag,
        "models": {},
    }

    # Floor: predict each code's training prevalence for every patient. Any TSTR
    # model that fails to beat this learned nothing from its synthetic data.
    y_true_val = np.stack([
        label_proc.process(r["rare_labels"]).numpy() for r in val_records
    ])
    prior = np.zeros(len(rare_pool))
    for rec in train_records:
        for code in rec["rare_labels"]:
            prior[label_proc.label_vocab[code]] += 1
    prior /= max(1, len(train_records))
    results["models"]["prior"] = score(
        y_true_val, np.tile(prior, (len(val_records), 1)), keep, ks)

    print("\n=== TRTR (ceiling) ===")
    y_true, y_prob = train_and_score(
        train_records, val_dataset, input_proc, label_proc, args, "trtr")
    results["models"]["trtr"] = score(y_true, y_prob, keep, ks)

    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run expects NAME=SAVE_DIR, got {spec!r}")
        name, save_dir = spec.split("=", 1)
        print(f"\n=== TSTR: {name} ===")
        synth = load_synthetic(save_dir)
        syn_records, syn_diag = build_records(synth, rare_set, f"synth-{name}")
        entry = {"diagnostics": syn_diag, "save_dir": save_dir}
        if syn_diag["n_distinct_rare_codes"] == 0 or not syn_records:
            # Reporting a number here would look like a score for a model that
            # never saw a positive. Say what actually happened instead.
            entry["degenerate"] = True
            entry["note"] = ("synthetic data contains no pooled rare codes; the "
                             "downstream model cannot learn the task")
            print(f"  [{name}] DEGENERATE: no rare codes in synthetic data")
        else:
            y_true, y_prob = train_and_score(
                syn_records, val_dataset, input_proc, label_proc, args,
                f"tstr:{name}")
            entry.update(score(y_true, y_prob, keep, ks))
            entry["degenerate"] = False
        results["models"][f"tstr:{name}"] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nSaved -> {args.out}\n")
    print(f"{'model':24s} {'pr_auc_macro':>13s} {'recall@10':>10s}")
    for name, res in results["models"].items():
        if res.get("degenerate"):
            print(f"{name:24s} {'DEGENERATE':>13s} {'-':>10s}")
        else:
            print(f"{name:24s} {res['pr_auc_macro']:13.4f} "
                  f"{res.get('recall_at_10', float('nan')):10.4f}")


if __name__ == "__main__":
    main()
