"""Test 2: machine-learning efficacy on pooled rare codes (TSTR).

The question: if a hospital trains a downstream model on the synthetic EHR its
regime produced, how well does that model predict which rare codes a real
patient carries? Train-on-Synthetic, Test-on-Real, specialised to the tail.

One classifier per arm
----------------------
Multi-generator regimes (``local``, ``fedavg_ft``) train **eight** classifiers,
one per hospital, each seeing only that hospital's synthetic data, and every one
is scored on the same pooled real validation set. Pooling the eight synthetic
sets before the downstream model would hand Local-Only exactly the cross-site
coverage federation is supposed to provide, at evaluation time, for free.

Single-generator regimes (``centralized``, ``fedavg``) train **one**. They have
one generator and one synthetic set, so there is no hospital identity to attach
to a slice; cutting that set into eight disjoint pieces would only add sampling
noise while pretending to be eight independent draws. Pass ``--train-budget`` to
hold the record count equal across regimes so the comparison is not decided by
how much synthetic data an arm happens to emit -- it defaults to 0 (uncapped),
which asks the different question of what each regime can do with everything it
has.

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

Which codes get scored
----------------------
``--code-pool`` picks the pool before any of the above happens:

* ``rare`` (default) -- the manifest's ``pooled_rare_codes``. The tail this
  test was built for, and the pool every result before 2026-09 used.
* ``common`` -- its complement. On a cohort whose rare threshold is loose
  (``rare_prevalence_max`` 0.05 catches almost every ICD code) this pool is
  nearly empty; check the printed size before reading anything into it.
* ``all`` -- both. Combined with ``--min-positives`` this is the useful
  selection: reject codes too thin to measure, then ``--n-eval-codes`` draws
  uniformly from what survives, so the head is filtered *in* rather than
  cherry-picked.

For ``common`` and ``all`` the support counts come from ``fold_support`` over
the evaluation fold, since the manifest carries them only for rare codes, and
the ``global_rare`` split is empty by construction -- every ``global_rare``
figure in the output is NaN.

**AP's floor moves with the pool.** A no-information ranker scores AP equal to
the base rate, so a well-supported pool starts near 0.03 where the tail starts
near 0.007. Compare arms to the ``prior`` row of the same file; an AP from one
pool means nothing against an AP from another.

Runs on one GPU. Loads eICU once and scores every regime, so submit it after the
training jobs finish -- it consumes the ``synthetic.json`` each run persists.

Example:
    python examples/fedpyhealth/test2_rare_efficacy.py \\
        --cohort-cache $FEDCOHORT_CACHE/strat8_random \\
        --run centralized=_outputs/centralized_E2_R20_strat8_utility_save \\
        --run local=_outputs/local_E2_R20_strat8_utility_save
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
from utils.cohort import (
    DEFAULT_CACHE_DIR,
    SUPPORT_BANDS,
    assign_folds,
    load_manifest,
    load_synthetic,
    load_shared_generator,
    fold_support,
    read_trajectories,
    sample_eval_codes,
    scored_codes,
)
SEED = 0

# SUPPORT_BANDS and assign_folds now live in utils/cohort.py so utils/eda/lengths.py
# can share them without importing torch. Imported above; unchanged in meaning.


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR,
                   help="cohort cache directory built by utils/cohort.py")
    p.add_argument("--run", action="append", default=[], metavar="NAME=SAVE_DIR",
                   help="a regime to score; SAVE_DIR is the run's _outputs/"
                        "<run_name>_save/ folder holding synthetic.json. "
                        "Repeatable.")
    p.add_argument("--fold", default="val", choices=["val", "test"],
                   help="real fold the classifiers are scored on (default: val "
                        "-- keep test held out until the final numbers)")
    p.add_argument("--min-positives", type=int, default=1,
                   help="a rare code needs this many positives in the pooled "
                        "scoring fold to be scored. Default 1 = score every "
                        "code that is scoreable at all; average_precision is "
                        "undefined at 0 positives, and low-support codes are "
                        "reported in their own support band rather than "
                        "dropped. NOTE this also controls what is MASKED out "
                        "of the classifier inputs, so raising it does not just "
                        "filter the report -- it changes the task")
    p.add_argument("--mask-folds", type=int, default=4,
                   help="partition scored codes into this many folds and strip "
                        "only one fold per model. K does not change coverage -- "
                        "every scored code sits in exactly one fold and so is "
                        "always tested -- it changes how much rare-to-rare "
                        "co-occurrence survives in each fold: K keeps (K-1)/K "
                        "of it, at K times the classifiers. 4 (75%, 168 "
                        "classifiers for 4 regimes) is the default; 1 strips "
                        "the whole tail at once and is smoke-only, since no arm "
                        "can learn co-occurrence that is not there")
    p.add_argument("--code-pool", default="rare",
                   choices=["rare", "common", "all"],
                   help="which codes to mask and score. 'rare' (default) is "
                        "the manifest's pooled-rare list, the tail this test "
                        "was built for. 'common' is its complement -- every "
                        "code in the fold that the cohort did NOT call rare -- "
                        "which trades the tail question for codes with enough "
                        "positives that AP and AUROC are actually estimable. "
                        "'all' pools both. For 'common'/'all' the support "
                        "counts are recomputed from the fold, since the "
                        "manifest only carries them for rare codes, and the "
                        "global_rare split is empty by construction")
    p.add_argument("--n-eval-codes", type=int, default=0,
                   help="DRAW MODE: instead of partitioning all scored codes "
                        "into --mask-folds folds, draw this many codes once, "
                        "strip only those, and train ONE classifier per arm "
                        "over them. Cuts classifiers by a factor of K (27 "
                        "instead of 108 at the default arm set) and leaves far "
                        "more rare-to-rare co-occurrence in the input (30 of "
                        "476 masked keeps 94%, against 75% at K=4). The cost "
                        "is statistical: the macro mean is an estimate over "
                        "the drawn codes, not a census, so run several "
                        "--eval-seed values and report the spread. 0 (default) "
                        "keeps the K-fold behaviour")
    p.add_argument("--eval-seed", type=int, default=0,
                   help="which draw --n-eval-codes takes. Changing it changes "
                        "the evaluation set, so results across seeds are "
                        "repeat measurements, not refinements")
    p.add_argument("--train-budget", type=int, default=0,
                   help="records per classifier. 0 (default) = UNCAPPED: every "
                        "arm trains on all the data it has, so a small "
                        "hospital's 79 real records are compared against the "
                        "thousands of synthetic ones a generator can produce. "
                        "That is the volume question. Pass an explicit N to cap "
                        "every arm at N instead, which asks the different "
                        "question of whether synthetic matches real "
                        "record-for-record")
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
    p.add_argument("--out",
                   default="_outputs/results/tests/test2_rare_efficacy.json")
    p.add_argument("--resume", action="store_true",
                   help="reuse per-classifier scores cached in <out>.partial "
                        "from an earlier run that was killed. The cache is "
                        "keyed by a signature covering the cohort, fold, "
                        "budget, run specs, mask folds and model "
                        "hyperparameters, so a cache from a different "
                        "configuration is ignored rather than mixed in")
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


SHARED_ARM_KEY = ""


def synthetic_arms(
    per_hospital: Dict[str, List[dict]], hospitals: Sequence[str], budget: int,
    shared: bool,
) -> Tuple[Dict[str, Dict[str, List[List[str]]]], bool]:
    """Turn a run's synthetic output into the classifier arms it supports.

    How many arms a regime yields is a property of the regime, not a knob:

    - ``centralized`` and ``fedavg`` train ONE generator, and ``train.py`` files
      its single output under every hospital key. That is **one** arm: one
      generator, one synthetic set, one classifier. Splitting it into eight
      disjoint draws would train eight classifiers on eight slices of the same
      distribution and score them against the same pooled validation set,
      manufacturing a spread that reads as cross-hospital variation when it is
      only generation sampling noise.
    - ``local`` and ``fedavg_ft`` train one generator per hospital, so each
      hospital is a genuinely distinct arm.

    Args:
        per_hospital: ``{hospital_id: [{patient_id, visits}, ...]}``.
        hospitals: Cohort hospital ids, in manifest order.
        budget: Records per classifier; ``<= 0`` means use every patient the
            generator produced, which is the default and the setting under
            which the "lots of synthetic data" claim is actually testable.
        shared: Whether ONE generator produced every hospital's set. This is
            read from the run (``load_shared_generator``) rather than inferred
            here: every generator numbers its output ``synthetic_0..N``
            independently, so eight different fine-tuned models emit identical
            *id* lists over different patients, and an id-based guess would
            collapse eight per-hospital classifiers into one.

    Returns:
        ``({arm_key: {patient_id: visits}}, shared)``. When ``shared`` is True
        the mapping holds exactly one entry, keyed ``SHARED_ARM_KEY``, because
        there is no hospital identity to attach to it.

    Raises:
        ValueError: If a generator cannot supply ``budget`` records.
    """
    def take(pool: List[dict], key: str, who: str
             ) -> Dict[str, List[List[str]]]:
        if budget > 0 and len(pool) < budget:
            raise ValueError(
                f"{who} has {len(pool)} synthetic patients but the budget is "
                f"{budget}. Raise --synth-per-hospital (multi-model regimes) or "
                "--num-synth (single-model regimes) on the run, lower "
                "--train-budget, or pass --train-budget 0 to use everything."
            )
        chosen = pool if budget <= 0 else pool[:budget]
        return {f"{key}:{p['patient_id']}": [list(v) for v in p["visits"]]
                for p in chosen}

    if shared:
        pool = per_hospital[hospitals[0]]
        return {SHARED_ARM_KEY: take(pool, "pooled", "the global generator")}, True

    return {hid: take(per_hospital.get(hid, []), hid, f"hospital {hid}")
            for hid in hospitals}, False


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
    """Average precision, AUROC and two F1 variants per code column.

    F1 needs a decision threshold, which AP and AUROC do not, and at these
    prevalences the threshold IS the metric: the rarest scored code appears in
    under 0.1% of patients, so no classifier here ever crosses 0.5 and a plain
    ``f1_score(y, p > 0.5)`` is 0.0 for every arm -- true, and useless for
    telling them apart. Two thresholds are reported instead, and they bracket
    the honest answer:

    ``f1_best``
        The maximum F1 over every threshold on the precision-recall curve. This
        is what a perfectly tuned operating point would give, and it is
        OPTIMISTICALLY BIASED -- the threshold is chosen using the same labels
        it is scored on, so it is an upper bound rather than an estimate of
        held-out performance. Comparable across arms (all get the same
        advantage), not quotable as "the F1 you would get".

    ``f1_prev``
        F1 at the threshold where the number of predicted positives equals the
        number of true positives. No label peeking beyond the positive COUNT,
        which is the standard unbiased choice for heavily imbalanced multilabel
        problems, and the number to quote.

    Both are ``nan`` for a one-class column, matching AP and AUROC: undefined,
    not zero.
    """
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
            prec, rec, _ = skm.precision_recall_curve(col_true, col_prob)
            denom = prec + rec
            f1s = np.where(denom > 0, 2 * prec * rec / np.maximum(denom, 1e-12), 0.0)
            entry["f1_best"] = float(np.max(f1s))
            # Predict exactly n_pos positives: take the n_pos highest scores.
            cut = np.partition(col_prob, -n_pos)[-n_pos]
            entry["f1_prev"] = float(skm.f1_score(col_true, col_prob >= cut,
                                                  zero_division=0))
        else:
            # One-class column: every one of these is undefined, not zero.
            entry["average_precision"] = float("nan")
            entry["roc_auc"] = float("nan")
            entry["f1_best"] = float("nan")
            entry["f1_prev"] = float("nan")
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
        def mean_of(key: str) -> float:
            vals = [scores[c][key] for c in codes
                    if key in scores[c] and not np.isnan(scores[c][key])]
            return float(np.mean(vals)) if vals else float("nan")

        aps = [scores[c]["average_precision"] for c in codes
               if not np.isnan(scores[c]["average_precision"])]
        return {
            "n_codes": len(codes),
            "n_scored": len(aps),
            "ap_macro": float(np.mean(aps)) if aps else float("nan"),
            "roc_auc_macro": mean_of("roc_auc"),
            "f1_best_macro": mean_of("f1_best"),
            "f1_prev_macro": mean_of("f1_prev"),
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


def run_signature(args, arms: Sequence[str], budget: int, n_folds: int) -> dict:
    """Everything that must match for a partial result to still be valid.

    A cached ``(arm, fold)`` score is only reusable if the task and the model
    that produced it are unchanged. Fold assignment depends on the scored code
    set (so on ``--min-positives`` and ``--fold``), the inputs depend on the
    budget and the run specs, and the numbers depend on the classifier
    hyperparameters. Any of those moving invalidates the cache -- silently
    mixing old and new scores in one table would be worse than recomputing.
    """
    return {
        "cohort_cache": args.cohort_cache,
        "fold": args.fold,
        "min_positives": args.min_positives,
        "mask_folds": n_folds,
        # Two draws of the same size share every other key here, so without
        # these a --resume would hand seed 0's cached scores to a seed 1 run and
        # report them as seed 1's. That failure is silent and the numbers look
        # entirely plausible, which makes it the worst kind.
        "n_eval_codes": args.n_eval_codes,
        "eval_seed": args.eval_seed,
        "train_budget": budget,
        "runs": sorted(args.run),
        "arms": sorted(arms),
        "model": {"epochs": args.epochs, "batch_size": args.batch_size,
                  "embedding_dim": args.embedding_dim,
                  "hidden_dim": args.hidden_dim, "lr": args.lr},
        "recall_at": args.recall_at,
    }


def load_partial(path: str, signature: dict) -> Dict[str, dict]:
    """Read cached ``(arm, fold)`` scores, ignoring any from a different setup."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        print(f"partial results at {path} unreadable ({e}); starting fresh")
        return {}
    if blob.get("signature") != signature:
        print(f"partial results at {path} were written for a different "
              "configuration; ignoring them and starting fresh")
        return {}
    return blob.get("entries", {})


def save_partial(path: str, signature: dict, entries: Dict[str, dict]) -> None:
    """Persist cached scores atomically (tmp + rename).

    Written after every single classifier. The whole point is surviving a wall
    -clock kill, so a half-written file at exactly the wrong moment would defeat
    the feature; rename is atomic, so the file on disk is always complete.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"signature": signature, "entries": entries}, fh)
    os.replace(tmp, path)


def spread(values: Sequence[float]) -> dict:
    """min / mean / median / max over a family's classifiers.

    Multi-model families (``local``, ``fedavg_ft``) have one classifier per
    hospital, so the spread is real cross-hospital variation. Single-model
    families (``centralized``, ``fedavg``) have exactly one, and ``n == 1``
    is the caller's cue to print a bare number instead of a fake interval.
    """
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
def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    ks = [int(k) for k in args.recall_at.split(",") if k.strip()]

    manifest = load_manifest(args.cohort_cache)
    hospitals = list(manifest["hospitals"])

    # Read the evaluation fold BEFORE choosing the pool: a non-rare pool has to
    # count its own support, which the manifest does not carry.
    eval_traj = read_trajectories(args.cohort_cache, args.fold, hospitals)

    if args.code_pool == "rare":
        strict = set(manifest.get("global_rare_codes", []))
        support: Dict[str, int] = manifest["pooled_rare_support"][args.fold]
        scored = scored_codes(manifest, args.fold, args.min_positives)
        pool_desc = f"pooled rare codes: {len(support)}"
    else:
        # No global_rare subset outside the rare list -- the manifest defines
        # that split only within it, so report it as empty rather than invent
        # one. Every `global_rare` figure below is NaN by construction here.
        strict = set()
        rare = set(manifest.get("pooled_rare_codes", []))
        seen = fold_support(eval_traj)
        pool = sorted(seen if args.code_pool == "all"
                      else (c for c in seen if c not in rare))
        support = {c: seen[c] for c in pool}
        scored = [c for c in pool if support[c] >= args.min_positives]
        pool_desc = (f"{args.code_pool} codes present in {args.fold}: "
                     f"{len(pool)} (of {len(seen)} in the fold, "
                     f"{len(rare)} rare)")
    if not scored:
        raise ValueError(
            f"no code in the {args.code_pool!r} pool has enough {args.fold} "
            "positives to score; lower --min-positives, or widen "
            "--rare-prevalence-max when building the cache"
        )
    dropped = len(support) - len(scored)
    print(f"{pool_desc}   scored (>= "
          f"{args.min_positives} {args.fold} positives): {len(scored)}   "
          f"dropped (< {args.min_positives} positives, unscoreable): {dropped}")
    if args.code_pool == "rare":
        print(f"  of the scored, globally rare (cohort prevalence <= "
              f"{manifest.get('global_rare_prevalence_max')}): "
              f"{len([c for c in scored if c in strict])}")

    # DRAW MODE. Narrowing `scored` is the whole change: assign_folds(scored, 1)
    # then yields a single fold holding exactly the drawn codes, and every
    # downstream stage -- masking, training, aggregation, banding, the
    # global_rare split -- already works off `scored` and needs no edit.
    pool_size = len(scored)
    if args.n_eval_codes > 0:
        if args.n_eval_codes > pool_size:
            raise SystemExit(
                f"--n-eval-codes {args.n_eval_codes} exceeds the {pool_size} "
                f"codes scoreable on {args.fold}; lower it or lower "
                "--min-positives."
            )
        scored = sample_eval_codes(scored, args.n_eval_codes, args.eval_seed)
        args.mask_folds = 1
        kept = 1 - len(scored) / pool_size
        band_counts: Dict[str, int] = {}
        for c in scored:
            for name, lo, hi in SUPPORT_BANDS:
                if lo <= support.get(c, 0) < hi:
                    band_counts[name] = 1 + band_counts.get(name, 0)
        print(f"\ndraw mode: {len(scored)} of {pool_size} codes, "
              f"seed {args.eval_seed}   ONE classifier per arm")
        print(f"  rare-to-rare co-occurrence left in the input: {kept:.0%} "
              f"(a {args.mask_folds}-of-{pool_size} mask, against 75% at K=4)")
        if strict:
            print(f"  drawn, globally rare: "
                  f"{len([c for c in scored if c in strict])}/{len(scored)}")
        print("  support bands: "
              + ", ".join(f"{k}={v}" for k, v in sorted(band_counts.items())))
        # 59% of this pool has 1-4 positives, so a faithful draw is mostly
        # low-support codes. That is a property of the pool, not of the draw --
        # but with 30 codes the noise no longer averages away as it does at 476.
        thin = sum(v for k, v in band_counts.items() if k == "1_4")
        if thin > len(scored) / 2:
            print(f"  NOTE: {thin}/{len(scored)} drawn codes have < 5 "
                  f"{args.fold} positives, so the macro mean rests mostly on "
                  "noisy per-code scores. Run several --eval-seed values.")

    # Straight off the cache: {hospital: {patient: [[code, ...], ...]}}.
    train_traj = read_trajectories(args.cohort_cache, "train", hospitals)
    real = {p: v for per in (train_traj, eval_traj)
            for pats in per.values() for p, v in pats.items()}
    train_by_hospital = {hid: sorted(pats) for hid, pats in train_traj.items()}
    val_ids = sorted(p for pats in eval_traj.values() for p in pats)
    # Which site each eval patient came from. The val set is POOLED -- every arm,
    # including the per-hospital ones, is scored on all eight sites' patients at
    # once -- so without this there is no way to ask how an arm does on its OWN
    # site as distinct from the cohort. Scoring a subset needs no retraining:
    # y_prob already covers every val row, so a per-site number is a row mask.
    val_hospital = {p: hid for hid, pats in eval_traj.items() for p in pats}
    all_train = sorted(p for ids in train_by_hospital.values() for p in ids)

    # budget 0 = UNCAPPED: every arm trains on everything it has. That is the
    # comparison this test exists for -- whether a lot of synthetic data beats
    # the little real data a small hospital actually holds. Capping every arm at
    # the smallest real split (the old default) answers a different question,
    # "is synthetic as good as real record-for-record", and makes the volume
    # claim untestable by construction, because volume is the variable.
    #
    # The cost is that arm sizes now differ by ~50x, so a TSTR win is a win on
    # DATA AVAILABLE, not on data quality. Say that when reporting it. Pass an
    # explicit --train-budget N to get the matched-budget comparison back.
    budget = args.train_budget
    uncapped = budget <= 0
    smallest_real = min(len(v) for v in train_by_hospital.values())
    if uncapped:
        print(f"per-classifier training budget: UNCAPPED -- every arm uses all "
              f"the data it has (smallest real train split is {smallest_real})")
    else:
        print(f"per-classifier training budget: {budget} records (explicit; "
              f"matched across every arm)")

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
    # Uncapped there is no record count to hold fixed, so the arm would be a
    # duplicate of real_pooled; it is skipped rather than reported twice.
    if uncapped:
        print("  real_pooled_budgeted: skipped (uncapped -- it would duplicate "
              "real_pooled)")
    else:
        arms["real_pooled_budgeted"] = budgeted(pooled_real, budget,
                                                seed=SEED + 99)
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
        built, shared = synthetic_arms(
            load_synthetic(save_dir), hospitals, budget,
            load_shared_generator(save_dir))
        regime_shared[regime] = shared
        note = ("one global generator -> 1 classifier" if shared
                else f"{len(built)} per-hospital generators -> "
                     f"{len(built)} classifiers")
        print(f"loaded synthetic for {regime}: {note}")
        for key, traj in built.items():
            name = f"tstr:{regime}" if shared else f"tstr:{regime}:{key}"
            arms[name] = traj
            arm_meta[name] = {"family": f"tstr:{regime}",
                              "hospital": None if shared else key,
                              "save_dir": save_dir}

    # Uncapped, how much data each arm got IS the independent variable, so it
    # travels with every result rather than living only in a log line.
    for name in arms:
        arm_meta[name]["n_train_records"] = len(arms[name])
    print("\nrecords per classifier:")
    for name in sorted(arms, key=lambda n: -len(arms[n])):
        print(f"  {name:34s} {len(arms[name]):>7d}")

    folds = assign_folds(scored, args.mask_folds)
    print(f"\nmask folds: {len(folds)} "
          f"({'strip-all' if len(folds) == 1 else 'fold-wise'}), "
          f"{[len(f) for f in folds]} codes each")
    n_classifiers = len(arms) * len(folds)
    print(f"classifiers to train: {len(arms)} arms x {len(folds)} folds = "
          f"{n_classifiers}")

    # test2 is the only stage measured in hours, so it is the only one where a
    # wall-clock kill can lose real work. Every finished classifier is written
    # to a sidecar immediately; --resume replays them instead of retraining.
    partial_path = args.out + ".partial"
    signature = run_signature(args, list(arms), budget, len(folds))
    partial = load_partial(partial_path, signature) if args.resume else {}
    if partial:
        print(f"resuming: {len(partial)}/{n_classifiers} classifiers already "
              f"done, from {partial_path}")
    elif args.resume:
        print(f"--resume set but no usable partial results at {partial_path}")

    # per-arm, per-code accumulators filled in across folds
    code_scores: Dict[str, Dict[str, dict]] = {a: {} for a in arms}
    # Parallel accumulators for the own-site view; empty for arms without one.
    own_code_scores: Dict[str, Dict[str, dict]] = {a: {} for a in arms}
    own_fold_recall: Dict[str, list] = {a: [] for a in arms}
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

            # One cache entry per trained classifier. Masking and the val
            # dataset are rebuilt on resume (cheap and deterministic); only the
            # training is skipped, which is the part measured in minutes.
            cache_key = f"{name}||{f_i}"
            cached = partial.get(cache_key)
            if cached is not None:
                code_scores[name].update(cached["code_scores"])
                fold_recall[name].append(cached["recall"])
                if cached.get("own_code_scores"):
                    own_code_scores[name].update(cached["own_code_scores"])
                    own_fold_recall[name].append(cached["own_recall"])
                print(f"  [{name}] fold {f_i}: cached, training skipped",
                      flush=True)
                continue

            y_prob = train_and_score(
                recs, val_dataset, input_proc, label_proc, args, name)
            scores = per_code_scores(y_true, y_prob, order)
            recall = {f"recall_at_{k}": recall_at_k(y_true, y_prob, k)
                      for k in ks}
            code_scores[name].update(scores)
            fold_recall[name].append(recall)

            # Own-site view, for arms that HAVE an own site. A per-hospital arm
            # is trained on one site but scored on all eight; restricting to its
            # own site's rows asks whether it serves the hospital that produced
            # it, which is the question a site actually cares about. Shared-
            # generator arms (fedavg, centralized, real_pooled, prior) have no
            # own site and are skipped rather than given a meaningless one.
            own_hid = arm_meta.get(name, {}).get("hospital")
            own_scores = own_recall = None
            if own_hid:
                rows = np.array([val_hospital.get(r["patient_id"]) == own_hid
                                 for r in val_records])
                if rows.sum() > 0:
                    own_scores = per_code_scores(
                        y_true[rows], y_prob[rows], order)
                    own_recall = {f"recall_at_{k}":
                                  recall_at_k(y_true[rows], y_prob[rows], k)
                                  for k in ks}
                    own_code_scores[name].update(own_scores)
                    own_fold_recall[name].append(own_recall)

            partial[cache_key] = {"code_scores": scores, "recall": recall,
                                  "own_code_scores": own_scores,
                                  "own_recall": own_recall}
            save_partial(partial_path, signature, partial)
            print(f"  [{name}] fold {f_i}: {len(partial)}/{n_classifiers} "
                  f"classifiers done", flush=True)

    # --- assemble results ---------------------------------------------------
    results = {
        "cohort_cache": args.cohort_cache,
        "cohort_name": manifest.get("cohort_name"),
        "fold": args.fold,
        "hospitals": hospitals,
        "min_positives": args.min_positives,
        "code_pool": args.code_pool,
        "n_pooled_rare_codes": len(support),
        "n_scored_codes": len(scored),
        "n_dropped_low_support": dropped,
        "mask_folds": len(folds),
        # Draw mode records the drawn codes themselves, not just the count: the
        # scored pool depends on the cohort cache, so (n, seed) alone does not
        # pin the subset if the cache is ever rebuilt.
        "n_eval_codes": args.n_eval_codes,
        "eval_seed": args.eval_seed,
        "eval_codes": list(scored) if args.n_eval_codes else None,
        "n_scoreable_pool": pool_size,
        "train_budget": budget,
        "train_budget_uncapped": uncapped,
        "smallest_real_train_split": smallest_real,
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
            # The same aggregation over this arm's OWN site's val rows only.
            # Nested under "own_site" rather than merged, so a consumer that
            # does not know about it cannot silently read a per-site number
            # where it expected the pooled one. Absent for arms with no own
            # site, which is how the page decides what it can plot.
            if own_code_scores.get(name):
                own = aggregate(own_code_scores[name], support, strict)
                for k in ks:
                    key = f"recall_at_{k}"
                    own[key] = float(np.mean(
                        [f[key] for f in own_fold_recall[name]
                         if not np.isnan(f[key])] or [np.nan]))
                own["n_val_patients"] = int(sum(
                    1 for h in val_hospital.values()
                    if h == arm_meta[name]["hospital"]))
                entry["own_site"] = own
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
            # Uncapped this is the independent variable, not a footnote.
            "n_train_records": spread(
                [float(arm_meta[n]["n_train_records"]) for n in names]),
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

    # The full results file supersedes the sidecar. Leaving it would invite a
    # later --resume to replay a stale cache that merely happens to match.
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"removed partial cache {partial_path}")

    # --- console table ------------------------------------------------------
    band_names = [b for b, _, _ in SUPPORT_BANDS]
    header = (f"\n{'family':22s} {'n':>3s} {'records':>15s} "
              f"{'AP overall':>18s} {'AP global-rare':>18s} " +
              " ".join(f"{'AP ' + b:>12s}" for b in band_names))
    print(header)
    print("-" * len(header))

    p = results["arms"]["prior"]
    print(f"{'prior (floor)':22s} {'-':>3s} {'-':>15s} "
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
        rec = e["n_train_records"]
        recs = (f"{int(rec['mean']):>15d}" if rec["min"] == rec["max"]
                else f"{int(rec['min']):>6d}-{int(rec['max']):<8d}")
        print(f"{family + deg:22s} {e['n_classifiers']:3d} {recs} "
              f"{fmt(e['overall_ap_macro'])} {fmt(e['global_rare_ap_macro'])} " +
              " ".join(f"{e['bands'][b]['mean']:12.4f}" for b in band_names))

    print("\nMulti-model families (local, fedavg_ft) show mean [min,max] across "
          "their per-hospital\nclassifiers. Single-model families (centralized, "
          "fedavg) are one generator -> one\nclassifier and show a bare number; "
          "the two spreads are not comparable quantities.")
    print("Low support bands (1_4, 5_9) are noisy per code -- read them with "
          "n_scored in hand,\nfrom the JSON, rather than as point estimates.")
    if uncapped:
        print(f"\nUNCAPPED: arms differ in training-set size (see the records "
              f"column), so a tstr\nwin over real_local is a win on DATA "
              f"AVAILABLE, not on per-record quality. That is\nthe intended "
              f"question -- whether a lot of synthetic data beats the "
              f"{smallest_real} real\nrecords a small hospital actually holds. "
              f"Report it as such.")
    print("\nThe claim to check: real_local < fedavg <= centralized <= real_pooled.")
    if len(folds) == 1 and not args.n_eval_codes:
        # Only a warning when the single fold holds the WHOLE pool. In draw mode
        # the single fold is 30 of 476 codes, which strips less of the tail than
        # K=4 does, so the co-occurrence objection does not apply.
        print("NOTE: --mask-folds 1 strips every scored rare code at once, so "
              "no rare-rare co-occurrence is available to any arm. Expect "
              "scores near the prior; use --mask-folds 4 for the headline run, "
              "or --n-eval-codes N to mask a small subset instead.")
    if args.n_eval_codes:
        print(f"\nNOTE: draw mode -- these numbers describe {len(scored)} "
              f"sampled codes, not all {pool_size}. They are NOT comparable to "
              "a --mask-folds 4 run (different mask size, different task "
              "difficulty). Compare draw-mode runs only against each other.")


if __name__ == "__main__":
    main()
