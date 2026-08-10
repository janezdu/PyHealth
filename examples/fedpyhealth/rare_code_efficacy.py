"""Test 2: machine-learning efficacy on pooled rare codes (TSTR).

The question: if a hospital trains a downstream model on the synthetic EHR its
regime produced, how well does that model predict which rare codes a real
patient carries? Train-on-Synthetic, Test-on-Real, specialised to the tail.

One classifier per hospital
---------------------------
Every arm trains **eight** classifiers, one per hospital, each seeing only that
hospital's data, and every one of them is scored on the same pooled real
validation set. Pooling the eight synthetic sets before the downstream model
would hand Local-Only exactly the cross-site coverage that federation is
supposed to provide, at evaluation time, for free -- and Local-Only would then
look competitive for reasons that have nothing to do with its generator.

Centralized and FedAvg produce a single global generator, so ``synthetic.json``
stores the same pooled output under all eight hospital keys. Those arms get
eight **disjoint** size-matched slices of it instead, which preserves the
protocol (8 classifiers, N samples each) and keeps sampling variance honest.

Every classifier trains on exactly ``--train-budget`` records. TSTR is strongly
sensitive to downstream training-set size, so without a fixed budget a regime
that happens to emit more synthetic patients wins on volume rather than on
fidelity.

The arms
--------
=====================  =====================================  ================
arm                    training data                          role
=====================  =====================================  ================
prior                  none (per-code training prevalence)    floor
real_local:<hid>       hospital <hid>'s REAL train split      the bar to beat
real_pooled_budgeted   all eight, cut to the same budget      diversity only
real_pooled            all eight real train splits            ceiling (TRTR)
tstr:<regime>:<hid>    hospital <hid>'s synthetic             the comparison
=====================  =====================================  ================

``real_pooled`` beats ``real_local`` for two reasons at once -- it sees more
records *and* records from more sites -- so on its own it cannot say which one
mattered. ``real_pooled_budgeted`` holds the record count at the per-hospital
budget and varies only site diversity, which splits the gap:
``real_local -> real_pooled_budgeted`` is what cross-site diversity buys, and
``real_pooled_budgeted -> real_pooled`` is what raw volume buys. Federation can
only ever deliver the first.

``real_local`` is the row that decides whether any of this is worth doing. If a
hospital's own real data beats every synthetic regime on the pooled validation
set, the generative pipeline buys nothing. The claim to land is
``real_local < fedavg <= centralized <= real_pooled``.

Masking
-------
A patient's input has rare codes stripped and must predict them back. Masking
only a patient's own positives would leave a "something was removed here"
asymmetry readable straight off the input, so within a fold the mask is applied
uniformly to every record in every arm.

``--mask-folds K`` controls how much of the tail is removed at once:

* ``K=1`` strips *all* scored rare codes from every input. Cheapest, but it also
  destroys rare-to-rare co-occurrence, which for tail codes is often the only
  signal there is -- the model is left inferring a 0.3%-prevalence code from
  common codes alone, and most arms collapse toward the prior.
* ``K>1`` partitions the scored codes into K disjoint folds and trains one model
  per fold, stripping only that fold's codes. Fold k keeps ``(K-1)/K`` of the
  rare co-occurrence structure. Costs K trainings per classifier.

Reading the numbers
-------------------
Three things are instrumented because they look like results but are not:

1. A generator that emits almost no rare codes trains a classifier on near
   all-zero targets, which collapses to the prior. That is a real finding, but
   it is indistinguishable from a plumbing bug without the ``diagnostics``
   block, so an arm with no rare codes in its synthetic data is reported as
   ``degenerate: true`` rather than as a score.
2. ``pooled_rare_codes`` mixes genuinely long-tail codes with codes that are
   rare at one site and common at another. Results are therefore reported both
   over the full scored pool and over ``global_rare_codes`` (the manifest's
   strict subset, cohort prevalence <= 1%). Do not average the two.
3. A macro mean over the scored pool is dominated by its head. Metrics are
   broken out by validation support band (5-9 / 10-19 / 20-49 / 50+); the
   federation claim lives in the 5-9 band and is invisible in the macro mean.

Runs on one GPU. Loads eICU once and scores every regime, so submit it after the
training jobs finish -- it consumes the ``synthetic.json`` each run persists.

Example:
    python examples/fedpyhealth/rare_code_efficacy.py \\
        --cohort-file examples/fedpyhealth/cohorts/rare8_v2.json \\
        --run centralized=_outputs/centralized_E2_R20_rare8_v2_utility_save \\
        --run local=_outputs/local_E2_R20_rare8_v2_utility_save
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

# Validation-support bands for stratified reporting. A single macro mean over
# the whole scored pool is separated mostly by its head, which is why the
# headline table reports these instead.
SUPPORT_BANDS: Tuple[Tuple[str, int, float], ...] = (
    ("5_9", 5, 10),
    ("10_19", 10, 20),
    ("20_49", 20, 50),
    ("50_plus", 50, float("inf")),
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-file",
                   default="examples/fedpyhealth/cohorts/rare8_v2.json",
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
    p.add_argument("--mask-folds", type=int, default=1,
                   help="partition scored codes into this many folds and strip "
                        "only one fold per model. 1 strips everything at once "
                        "(cheap, destroys rare-rare co-occurrence); 10 is the "
                        "recommended setting for a headline run and costs 10x")
    p.add_argument("--train-budget", type=int, default=0,
                   help="records per classifier. 0 = the smallest hospital's "
                        "real train split, which is the largest budget every "
                        "arm can actually meet")
    p.add_argument("--pooled-budget", type=int, default=0,
                   help="records for the real_pooled ceiling. 0 = all of them "
                        "(the ceiling is supposed to show what pooling buys)")
    p.add_argument("--skip-real-local", action="store_true",
                   help="skip the 8 real_local classifiers (saves time on a "
                        "re-score where that baseline is already known)")
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
    visits: Sequence[Sequence[str]], masked_codes: Set[str]
) -> Tuple[List[List[str]], List[str]]:
    """Strip ``masked_codes`` from the input and keep them as the label.

    Codes are de-duplicated within a visit, first occurrence kept. eICU charts
    the same diagnosis repeatedly through a unit stay, so raw visits average 28
    codes and run to 3601, while the distinct set averages 4.7 and tops out at
    67. The repetition is a charting artifact, not clinical signal -- the task
    here is set membership ("does this patient carry the code"), the padded
    tensor shrinks by a factor of ~45, and visits stop overflowing the inner
    length the input processor was fitted with.

    Args:
        visits: The patient's trajectory as lists of code strings.
        masked_codes: Every code this fold removes -- not just this patient's.
            Applying the mask uniformly is what stops the model from reading
            "a code was deleted here" off the input.

    Returns:
        ``(masked_visits, labels)``. Emptied visits are dropped; a patient whose
        codes are *all* masked ends up with no visits and must be discarded by
        the caller (there is nothing left to predict from).
    """
    masked, labels = [], set()
    for visit in visits:
        kept = []
        seen = set()
        for code in visit:
            if code in masked_codes:
                labels.add(code)
            elif code not in seen:
                seen.add(code)
                kept.append(code)
        if kept:
            masked.append(kept)
    return masked, sorted(labels)


def truncate_visits(records: List[dict], max_len: int) -> int:
    """Cap every visit at ``max_len`` codes, in place. Returns visits touched.

    ``NestedSequenceProcessor.process`` pads a visit up to the inner length it
    was fitted with but never truncates one that is longer, which yields a
    ragged list and a ``torch.tensor`` failure. Since the processors here are
    deliberately fitted once on real pooled train and then reused for every
    arm, a synthetic generator emitting a longer visit than anything in real
    training would otherwise crash the run. Capping to the fitted length makes
    that impossible by construction, and is applied to every arm identically.
    """
    touched = 0
    for rec in records:
        for i, visit in enumerate(rec["visits"]):
            if len(visit) > max_len:
                rec["visits"][i] = visit[:max_len]
                touched += 1
    return touched


def build_records(
    trajectories: Dict[str, List[List[str]]], masked_codes: Set[str],
) -> Tuple[List[dict], dict]:
    """Turn ``{patient_id: visits}`` into masked multilabel samples.

    Returns:
        ``(records, diagnostics)``. Diagnostics report how much of the input
        survived masking and how much rare-code signal is present -- the numbers
        that tell a real result apart from a degenerate one.
    """
    records, dropped = [], 0
    n_with_label, label_counts, distinct = 0, [], set()
    for pid, visits in trajectories.items():
        masked, labels = mask_and_label(visits, masked_codes)
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


def load_synthetic(save_dir: str) -> Dict[str, List[dict]]:
    """Read a run's persisted per-hospital synthetic patients."""
    path = os.path.join(save_dir, "synthetic.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is written at the end of ehr_eicu.py; a run "
            "that timed out before STEP 9 has no synthetic data to score."
        )
    with open(path) as fh:
        blob = json.load(fh)
    if "per_hospital" not in blob:
        raise ValueError(
            f"{path} has no 'per_hospital' block -- it predates the per-hospital "
            "classifier protocol. Re-run the regime, or score it with an older "
            "revision of this script."
        )
    return blob["per_hospital"]


def slice_per_hospital(
    per_hospital: Dict[str, List[dict]], hospitals: Sequence[str], budget: int,
) -> Tuple[Dict[str, Dict[str, List[List[str]]]], bool]:
    """Give every hospital ``budget`` synthetic patients, disjointly if shared.

    Centralized and FedAvg train one global generator, and ``ehr_eicu.py`` files
    the same pooled output under every hospital key. Training eight identical
    classifiers on it would report eight identical numbers and a spread of zero.
    Those arms instead get disjoint consecutive slices, so the protocol still
    reads "8 classifiers, ``budget`` records each" and the spread reflects real
    sampling variance.

    Args:
        per_hospital: ``{hospital_id: [{patient_id, visits}, ...]}``.
        hospitals: Cohort hospital ids, in manifest order.
        budget: Records per hospital.

    Returns:
        ``({hospital_id: {patient_id: visits}}, shared)`` where ``shared`` says
        whether the input was one global set rather than eight local ones.

    Raises:
        ValueError: If a hospital cannot supply ``budget`` records, or if the
            shared global set is too small to slice disjointly.
    """
    ids = {h: [str(p["patient_id"]) for p in per_hospital.get(h, [])]
           for h in hospitals}
    shared = len({tuple(v) for v in ids.values()}) == 1 and len(hospitals) > 1

    out: Dict[str, Dict[str, List[List[str]]]] = {}
    if shared:
        pool = per_hospital[hospitals[0]]
        need = budget * len(hospitals)
        if len(pool) < need:
            raise ValueError(
                f"one global synthetic set of {len(pool)} patients cannot give "
                f"{len(hospitals)} hospitals {budget} disjoint records each "
                f"({need} needed). Raise --num-synth on the run, or lower "
                "--train-budget."
            )
        for i, hid in enumerate(hospitals):
            chunk = pool[i * budget:(i + 1) * budget]
            out[hid] = {f"{hid}:{p['patient_id']}": [list(v) for v in p["visits"]]
                        for p in chunk}
        return out, True

    for hid in hospitals:
        pool = per_hospital.get(hid, [])
        if len(pool) < budget:
            raise ValueError(
                f"hospital {hid} has {len(pool)} synthetic patients but the "
                f"budget is {budget}. Raise --num-synth on the run, or lower "
                "--train-budget."
            )
        out[hid] = {f"{hid}:{p['patient_id']}": [list(v) for v in p["visits"]]
                    for p in pool[:budget]}
    return out, False


def budgeted(
    trajectories: Dict[str, List[List[str]]], budget: int, seed: int,
) -> Dict[str, List[List[str]]]:
    """Take ``budget`` patients without replacement, deterministically."""
    pids = sorted(trajectories)
    if budget <= 0 or budget >= len(pids):
        return {p: trajectories[p] for p in pids}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pids), size=budget, replace=False)
    return {pids[int(i)]: trajectories[pids[int(i)]] for i in sorted(chosen)}


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def make_dataset(records, input_proc, label_proc):
    """Build a SampleDataset that reuses already-fitted processors.

    Passing pre-fitted processors makes ``SampleBuilder.fit`` skip fitting, so
    the code vocabulary and label space are identical across every arm inside a
    fold. Without that, each classifier would be scored against a different
    label axis.
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
) -> np.ndarray:
    """Train one RNN on ``train_records`` and return its validation scores."""
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
    _, y_prob, _ = trainer.inference(
        get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    )
    print(f"  [{tag}] trained on {len(train_records)} records", flush=True)
    return np.asarray(y_prob)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Patient-averaged fraction of true rare codes inside the top-k predicted.

    Only patients with at least one positive contribute; the metric is
    undefined for the rest. With ``--mask-folds K > 1`` this ranks within a
    fold, not across the whole pool, so it is a per-fold diagnostic rather than
    a patient-level ranking of the entire tail.
    """
    scores = []
    order = np.argsort(-y_prob, axis=1)[:, :k]
    for row, top in zip(y_true, order):
        n_pos = int(row.sum())
        if n_pos == 0:
            continue
        scores.append(float(row[top].sum()) / min(n_pos, k))
    return float(np.mean(scores)) if scores else float("nan")


def per_code_scores(
    y_true: np.ndarray, y_prob: np.ndarray, codes: Sequence[str]
) -> Dict[str, dict]:
    """Average precision and AUROC for each code column independently."""
    from sklearn import metrics as skm

    out = {}
    for j, code in enumerate(codes):
        col_true, col_prob = y_true[:, j], y_prob[:, j]
        n_pos = int(col_true.sum())
        entry = {"n_val_positives": n_pos}
        if 0 < n_pos < len(col_true):
            entry["average_precision"] = float(
                skm.average_precision_score(col_true, col_prob))
            entry["roc_auc"] = float(skm.roc_auc_score(col_true, col_prob))
        else:
            # One-class column: both metrics are undefined, not zero.
            entry["average_precision"] = float("nan")
            entry["roc_auc"] = float("nan")
        out[code] = entry
    return out


def aggregate(
    scores: Dict[str, dict], support: Dict[str, int], strict: Set[str],
) -> dict:
    """Collapse per-code scores into band, strict-pool and overall means.

    Args:
        scores: ``{code: {"average_precision": ..., "roc_auc": ...}}``.
        support: Validation positives per code, from the manifest.
        strict: The ``global_rare_codes`` subset (cohort prevalence <= 1%).

    Returns:
        Macro means over the whole scored pool, over the strict pool, and over
        each validation-support band, each with the code count behind it.
    """
    def macro(codes: Sequence[str]) -> dict:
        aps = [scores[c]["average_precision"] for c in codes
               if not np.isnan(scores[c]["average_precision"])]
        aucs = [scores[c]["roc_auc"] for c in codes
                if not np.isnan(scores[c]["roc_auc"])]
        return {
            "n_codes": len(codes),
            "n_scored": len(aps),
            "ap_macro": float(np.mean(aps)) if aps else float("nan"),
            "roc_auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
        }

    all_codes = sorted(scores)
    out = {
        "overall": macro(all_codes),
        "global_rare": macro([c for c in all_codes if c in strict]),
        "bands": {},
    }
    for name, lo, hi in SUPPORT_BANDS:
        band = [c for c in all_codes if lo <= support.get(c, 0) < hi]
        out["bands"][name] = macro(band)
    return out


def spread(values: Sequence[float]) -> dict:
    """min / mean / median / max over the eight per-hospital classifiers."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"min": float("nan"), "mean": float("nan"),
                "median": float("nan"), "max": float("nan"), "n": 0}
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def assign_folds(codes: Sequence[str], n_folds: int) -> List[List[str]]:
    """Round-robin the scored codes into ``n_folds`` disjoint, sorted folds.

    Round-robin over the *sorted* code list is deterministic and spreads
    prevalence evenly, so no fold ends up holding only the head or only the
    2-positive tail.
    """
    folds: List[List[str]] = [[] for _ in range(max(1, n_folds))]
    for i, code in enumerate(sorted(codes)):
        folds[i % len(folds)].append(code)
    return [f for f in folds if f]


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
    meta = manifest["meta"]
    hospitals = [h["hospital_id"] for h in manifest["hospitals"]]
    strict = set(meta.get("global_rare_codes", []))
    support: Dict[str, int] = meta.get("pooled_rare_val_support", {})
    if not support:
        raise ValueError(
            "manifest has no 'pooled_rare_val_support'; re-run "
            "freeze_cohort_split.py so support bands can be frozen with the "
            "split rather than re-derived here."
        )

    scored = sorted(c for c, n in support.items() if n >= args.min_positives)
    if not scored:
        raise ValueError(
            "no rare code has enough validation positives to score; lower "
            "--min-positives or widen --rare-prevalence-max in the freeze step"
        )
    dropped = len(support) - len(scored)
    print(f"pooled rare codes: {len(support)}   scored (>= "
          f"{args.min_positives} val positives): {len(scored)}   "
          f"dropped: {dropped}")
    print(f"  of the scored, globally rare (cohort prevalence <= "
          f"{meta.get('global_rare_prevalence_max')}): "
          f"{len([c for c in scored if c in strict])}")

    train_by_hospital = {h["hospital_id"]: list(h["train_patient_ids"])
                         for h in manifest["hospitals"]}
    val_ids = sorted({p for h in manifest["hospitals"]
                      for p in h["val_patient_ids"]})
    all_train = sorted({p for ids in train_by_hospital.values() for p in ids})

    budget = args.train_budget or min(len(v) for v in train_by_hospital.values())
    print(f"per-classifier training budget: {budget} records "
          f"({'explicit' if args.train_budget else 'smallest real train split'})")

    real = load_real_trajectories(
        args.eicu_root, set(all_train) | set(val_ids), dev=args.dev
    )

    # --- assemble every arm's raw trajectories (masking happens per fold) ---
    arms: Dict[str, Dict[str, List[List[str]]]] = {}
    arm_meta: Dict[str, dict] = {}

    pooled_budget = args.pooled_budget or len(all_train)
    pooled_real = {p: real[p] for p in all_train}
    arms["real_pooled"] = budgeted(pooled_real, pooled_budget, seed=SEED)
    arm_meta["real_pooled"] = {"family": "real_pooled", "hospital": None}

    # real_pooled sees every hospital's data, so it is better in two distinct
    # ways at once: more records AND records from more sites. Comparing it
    # straight against real_local cannot say which one mattered. This arm holds
    # the record count fixed at the per-hospital budget and varies only the
    # diversity, splitting the gap into
    #   real_local -> real_pooled_budgeted   what cross-site diversity buys
    #   real_pooled_budgeted -> real_pooled  what raw volume buys
    arms["real_pooled_budgeted"] = budgeted(pooled_real, budget, seed=SEED + 99)
    arm_meta["real_pooled_budgeted"] = {"family": "real_pooled_budgeted",
                                        "hospital": None}

    if not args.skip_real_local:
        for i, hid in enumerate(hospitals):
            name = f"real_local:{hid}"
            arms[name] = budgeted(
                {p: real[p] for p in train_by_hospital[hid]}, budget, seed=SEED + i)
            arm_meta[name] = {"family": "real_local", "hospital": hid}

    regime_shared: Dict[str, bool] = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run expects NAME=SAVE_DIR, got {spec!r}")
        regime, save_dir = spec.split("=", 1)
        sliced, shared = slice_per_hospital(
            load_synthetic(save_dir), hospitals, budget)
        regime_shared[regime] = shared
        note = "one global set, sliced disjointly" if shared else "per-hospital"
        print(f"loaded synthetic for {regime}: {note}")
        for hid in hospitals:
            name = f"tstr:{regime}:{hid}"
            arms[name] = sliced[hid]
            arm_meta[name] = {"family": f"tstr:{regime}", "hospital": hid,
                              "save_dir": save_dir}

    folds = assign_folds(scored, args.mask_folds)
    print(f"mask folds: {len(folds)} "
          f"({'strip-all' if len(folds) == 1 else 'fold-wise'}), "
          f"{[len(f) for f in folds]} codes each")
    print(f"classifiers to train: {len(arms)} arms x {len(folds)} folds = "
          f"{len(arms) * len(folds)}")

    # per-arm, per-code accumulators filled in across folds
    code_scores: Dict[str, Dict[str, dict]] = {a: {} for a in arms}
    fold_recall: Dict[str, List[Dict[str, float]]] = {a: [] for a in arms}
    diagnostics: Dict[str, dict] = {}
    degenerate: Dict[str, bool] = {a: True for a in arms}
    prior_scores: Dict[str, dict] = {}

    for f_i, fold_codes in enumerate(folds):
        mask = set(fold_codes)
        print(f"\n=== fold {f_i + 1}/{len(folds)}: {len(fold_codes)} codes ===",
              flush=True)

        val_records, val_diag = build_records(
            {p: real[p] for p in val_ids}, mask)
        train_records_by_arm = {}
        for name, traj in arms.items():
            recs, diag = build_records(traj, mask)
            train_records_by_arm[name] = recs
            if diag["n_distinct_rare_codes"] > 0 and recs:
                degenerate[name] = False
            diagnostics.setdefault(name, {})[f"fold_{f_i}"] = diag
        print(f"  val: {val_diag}", flush=True)

        # Freeze the feature space on the pooled real training data. Codes a
        # generator invents that were never seen in real training map to <unk>,
        # which is the correct TSTR behaviour.
        input_proc = NestedSequenceProcessor()
        input_proc.fit(train_records_by_arm["real_pooled"], "visits")
        max_inner = input_proc.size()
        clipped = truncate_visits(val_records, max_inner)
        for recs in train_records_by_arm.values():
            clipped += truncate_visits(recs, max_inner)
        if clipped:
            print(f"  truncated {clipped} visits to the fitted inner length "
                  f"{max_inner}", flush=True)

        label_proc = MultiLabelProcessor()
        label_proc.fit([{"rare_labels": fold_codes}], "rare_labels")
        val_dataset = make_dataset(val_records, input_proc, label_proc)

        order = [None] * len(fold_codes)
        for code in fold_codes:
            order[label_proc.label_vocab[code]] = code
        y_true = np.stack([
            label_proc.process(r["rare_labels"]).numpy() for r in val_records
        ])

        # Floor: predict each code's pooled-real-train prevalence for everyone.
        # Any classifier that fails to beat this learned nothing.
        prior = np.zeros(len(fold_codes))
        pooled_train = train_records_by_arm["real_pooled"]
        for rec in pooled_train:
            for code in rec["rare_labels"]:
                prior[label_proc.label_vocab[code]] += 1
        prior /= max(1, len(pooled_train))
        prior_scores.update(per_code_scores(
            y_true, np.tile(prior, (len(val_records), 1)), order))

        for name in arms:
            recs = train_records_by_arm[name]
            diag = diagnostics[name][f"fold_{f_i}"]
            if not recs or diag["n_distinct_rare_codes"] == 0:
                # Scoring here would look like a result for a model that never
                # saw a positive. Leave the codes unscored instead.
                print(f"  [{name}] fold {f_i}: no rare codes in training data, "
                      "skipped", flush=True)
                continue
            y_prob = train_and_score(
                recs, val_dataset, input_proc, label_proc, args, name)
            code_scores[name].update(per_code_scores(y_true, y_prob, order))
            fold_recall[name].append(
                {f"recall_at_{k}": recall_at_k(y_true, y_prob, k) for k in ks})

    # --- assemble results ---------------------------------------------------
    results = {
        "cohort_file": args.cohort_file,
        "cohort_name": meta.get("cohort_name"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "hospitals": hospitals,
        "min_positives": args.min_positives,
        "n_pooled_rare_codes": len(support),
        "n_scored_codes": len(scored),
        "n_dropped_low_support": dropped,
        "mask_folds": len(folds),
        "train_budget": budget,
        "pooled_budget": pooled_budget,
        "regime_shared_generator": regime_shared,
        "arms": {},
        "families": {},
    }

    results["arms"]["prior"] = {
        "family": "prior", "hospital": None, "degenerate": False,
        **aggregate(prior_scores, support, strict),
    }
    for name in arms:
        entry = dict(arm_meta[name])
        entry["degenerate"] = degenerate[name]
        entry["diagnostics"] = diagnostics.get(name, {})
        if degenerate[name] or not code_scores[name]:
            entry["note"] = ("training data contains no scored rare codes; the "
                             "downstream model cannot learn the task")
        else:
            entry.update(aggregate(code_scores[name], support, strict))
            for k in ks:
                key = f"recall_at_{k}"
                entry[key] = float(np.mean(
                    [f[key] for f in fold_recall[name] if not np.isnan(f[key])]
                    or [np.nan]))
        results["arms"][name] = entry

    # Collapse the eight per-hospital classifiers of each family into a spread.
    by_family: Dict[str, List[str]] = {}
    for name, m in arm_meta.items():
        by_family.setdefault(m["family"], []).append(name)
    for family, names in by_family.items():
        live = [n for n in names if not degenerate[n] and code_scores[n]]
        entry = {
            "n_classifiers": len(names),
            "n_degenerate": len(names) - len(live),
            "overall_ap_macro": spread(
                [results["arms"][n]["overall"]["ap_macro"] for n in live]),
            "global_rare_ap_macro": spread(
                [results["arms"][n]["global_rare"]["ap_macro"] for n in live]),
            "bands": {},
        }
        for band, _, _ in SUPPORT_BANDS:
            entry["bands"][band] = spread(
                [results["arms"][n]["bands"][band]["ap_macro"] for n in live])
        results["families"][family] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {args.out}")

    # --- console table ------------------------------------------------------
    band_names = [b for b, _, _ in SUPPORT_BANDS]
    header = (f"\n{'family':22s} {'n':>3s} {'AP overall':>18s} "
              f"{'AP global-rare':>18s} " +
              " ".join(f"{'AP ' + b:>12s}" for b in band_names))
    print(header)
    print("-" * len(header))

    p = results["arms"]["prior"]
    print(f"{'prior (floor)':22s} {'-':>3s} "
          f"{p['overall']['ap_macro']:18.4f} "
          f"{p['global_rare']['ap_macro']:18.4f} " +
          " ".join(f"{p['bands'][b]['ap_macro']:12.4f}" for b in band_names))

    def fmt(sp: dict) -> str:
        if sp["n"] == 0:
            return f"{'-':>18s}"
        if sp["n"] == 1:
            return f"{sp['mean']:18.4f}"
        return f"{sp['mean']:8.4f} [{sp['min']:.3f},{sp['max']:.3f}]"

    order = (["real_local", "real_pooled_budgeted", "real_pooled"]
             + [f for f in results["families"] if f.startswith("tstr:")])
    for family in order:
        if family not in results["families"]:
            continue
        e = results["families"][family]
        deg = f" ({e['n_degenerate']} degenerate)" if e["n_degenerate"] else ""
        print(f"{family + deg:22s} {e['n_classifiers']:3d} "
              f"{fmt(e['overall_ap_macro'])} {fmt(e['global_rare_ap_macro'])} " +
              " ".join(f"{e['bands'][b]['mean']:12.4f}" for b in band_names))

    print("\nPer-hospital classifiers show mean [min,max] across the cohort.")
    print("The claim to check: real_local < fedavg <= centralized <= real_pooled.")
    if len(folds) == 1:
        print("NOTE: --mask-folds 1 strips every scored rare code at once, so "
              "no rare-rare co-occurrence is available to any arm. Expect "
              "scores near the prior; use --mask-folds 10 for the headline run.")


if __name__ == "__main__":
    main()
