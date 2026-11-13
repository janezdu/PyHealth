"""Merge sharded test2 runs into one file per (classifier, eval seed).

Sharding is by ``--run`` arm, which is safe because the scored code set comes
from the manifest and the real evaluation fold alone, never from ``--run``
(``test2_rare_efficacy.py`` lines 975-1034). This script re-checks that
invariant rather than trusting it: if two shards disagree on ``eval_codes``,
``fold``, ``min_positives`` or ``train_budget`` it refuses to merge, because
numbers scored against different code sets are not comparable.

The merged file carries the union of ``arms`` and sets ``families`` to null.
The per-family min/mean/median/max block is NOT recomputed -- reproducing that
aggregation here would mean a second implementation that could silently drift
from the scorer's. Read families from ``arms`` instead, which is what the
plotting and table scripts already do.

Usage::

    python examples/fedpyhealth/scripts/merge_test2_shards.py --seed 1
    python examples/fedpyhealth/scripts/merge_test2_shards.py --seed 1 --seed 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

RES = "_outputs/results/tests"
# Everything here must agree across shards or the merge is meaningless.
MUST_MATCH = ("cohort_name", "fold", "min_positives", "classifier",
              "code_pool", "eval_seed", "n_eval_codes", "eval_codes",
              "train_budget", "n_scoreable_pool", "rlsp")


def merge_seed(clf: str, seed: int, out_path: str | None = None) -> str | None:
    """Merge every shard found for one classifier and eval seed.

    Args:
        clf: Classifier name as it appears in the filename, e.g. ``"gbm"``.
        seed: The ``--eval-seed`` the shards were scored at.
        out_path: Where to write. Defaults to the unsharded filename, so
            downstream scripts need no change.

    Returns:
        The path written, or None if no shard was found.

    Raises:
        SystemExit: If the shards disagree on any field in ``MUST_MATCH``.
    """
    pat = f"{RES}/test2_clf_{clf}_seed{seed}_shard*.json"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no shards matching {pat}")
        return None

    base: Dict | None = None
    arms: Dict[str, Dict] = {}
    overlap: List[str] = []
    for f in files:
        d = json.load(open(f))
        if base is None:
            base = d
        else:
            for k in MUST_MATCH:
                if d.get(k) != base.get(k):
                    raise SystemExit(
                        f"{f} disagrees with {files[0]} on {k!r}:\n"
                        f"  {files[0]}: {base.get(k)!r}\n  {f}: {d.get(k)!r}\n"
                        "Refusing to merge -- these were not scored against "
                        "the same codes."
                    )
        for name, v in d["arms"].items():
            # prior / real_pooled / real_local recur in shard 1 only, but guard
            # anyway: identical inputs give identical results, so a clash is a
            # sign something differed.
            if name in arms and arms[name] != v:
                overlap.append(name)
            arms[name] = v
        print(f"  {os.path.basename(f)}: {len(d['arms'])} arms")

    if overlap:
        print(f"  !! {len(overlap)} arm(s) appeared in more than one shard "
              f"with DIFFERENT results, last one kept: {sorted(set(overlap))}")

    assert base is not None
    merged = dict(base)
    merged["arms"] = arms
    merged["families"] = None   # see module docstring
    merged["merged_from"] = [os.path.basename(f) for f in files]
    out = out_path or f"{RES}/test2_clf_{clf}_seed{seed}.json"
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists -- move it aside first")
    with open(out, "w") as fh:
        json.dump(merged, fh)
    print(f"-> {out}   {len(arms)} arms from {len(files)} shard(s)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clf", default="gbm", help="classifier name in filenames")
    p.add_argument("--seed", type=int, action="append", required=True,
                   help="eval seed to merge; repeatable")
    p.add_argument("--out", default=None,
                   help="output path (only valid with a single --seed)")
    args = p.parse_args()
    if args.out and len(args.seed) > 1:
        raise SystemExit("--out takes a single --seed")
    for s in args.seed:
        print(f"seed {s}:")
        merge_seed(args.clf, s, args.out)


if __name__ == "__main__":
    main()
