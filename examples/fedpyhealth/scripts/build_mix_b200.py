"""Rebuild mix_spec/mix_gen for the unweighted frontier, from the b200 files.

The frontier figure (scripts/plot_unweighted.sh) reads a merged pair,
``mix_spec.json`` / ``mix_gen.json``. The previous pair was assembled by hand
and carried NO metadata -- no cohort, fold, real_scope, rare_scope, synth_cap
or n_bootstraps -- so there was no way to tell what settings produced it, and
its numbers disagreed with every ``test1_*_b200.json`` on every shared arm
(fedavg's specialist R^2 moved 0.55, local's generalist 0.60). Most likely it
predates the move to 200 bootstraps, where Test 1 differences were inside the
noise floor.

This script rebuilds it from the b200 files only, copies the provenance
forward, and refuses to merge sources that disagree on how they were scored.

WHAT THIS FIGURE IS. Every arm is UNWEIGHTED except centralized, which is
borrowed from the reweighted family as the pooled-data anchor -- the unweighted
centralized sits at (-44, -53) on all codes, far off any readable axis. That
borrowing is the whole reason the arm is named ``centralized_rw`` rather than
``centralized``, and why the figure's title says so.

Run:
    .venv/bin/python examples/fedpyhealth/scripts/build_mix_b200.py
"""
from __future__ import annotations

import json
import os
import shutil
import time

T = "_outputs/results/tests"

#: (arm in the output, arm in the source, source stem, legend note).
#:
#: Order matters only for readability; every arm names its own source, so
#: nothing here depends on a first-file-wins rule.
#:
#: fedavg_ft is the one judgement call. The old mix had it at specialist
#: +0.153, which matches ``full_ft`` in the NON-b200 test1_adapters file. At
#: b200 there is no 2-epoch full fine-tune -- the only unweighted full-FT arm
#: scored at 200 bootstraps is ``full_ft20``, which fine-tunes for 20 epochs.
#: That is a different arm, not the same arm re-scored, so it is labelled as
#: such rather than quietly standing in for the old point.
SOURCES = [
    ("local",          "local",         "baselines", ""),
    ("fedavg",         "fedavg",        "baselines", ""),
    ("centralized_rw", "centralized",   "rwfig",     "reweighted, borrowed"),
    ("fedavg_ft",      "full_ft20",     "adapters20", "20 fine-tune epochs"),
    ("fedavg_irm1e4",  "fedavg_irm1e4", "baselines", ""),
    ("fed_xm4_do03",   "fed_xm4_do03",  "fedxm4",    ""),
    ("do01_fed_irm",   "do01_fed_irm",  "xmirm",     ""),
]

#: Scoring settings that must agree across every merged file. real_scope is
#: deliberately absent: it is 'hospital' in the spec files and 'pooled' in the
#: gen files by construction, which is the whole point of the two axes.
MUST_MATCH = ("cohort_name", "fold", "rare_scope", "synth_cap", "n_bootstraps")


def path(stem: str, axis: str) -> str:
    return os.path.join(T, f"test1_{stem}_{axis}_b200.json")


def build(axis: str) -> dict:
    """Merge one axis ('spec' or 'gen') across the source files."""
    loaded, out_runs, provenance = {}, {}, {}
    settings = None
    for out_arm, src_arm, stem, note in SOURCES:
        p = path(stem, axis)
        if stem not in loaded:
            if not os.path.exists(p):
                raise SystemExit(f"missing source file: {p}")
            loaded[stem] = json.load(open(p))
        blob = loaded[stem]
        here = {k: blob.get(k) for k in MUST_MATCH}
        if settings is None:
            settings = here
        elif here != settings:
            # Merging files scored differently produces a figure where a gap
            # between two points can be the settings rather than the arms.
            diff = {k: (settings[k], here[k]) for k in MUST_MATCH
                    if settings[k] != here[k]}
            raise SystemExit(
                f"{os.path.basename(p)} was scored differently from the "
                f"earlier sources: {diff}. Refusing to merge -- re-score it "
                f"to match before rebuilding."
            )
        runs = blob.get("runs", {})
        if src_arm not in runs:
            raise SystemExit(
                f"{os.path.basename(p)} has no arm {src_arm!r}; it holds "
                f"{sorted(runs)}"
            )
        out_runs[out_arm] = runs[src_arm]
        provenance[out_arm] = {
            "source_file": os.path.basename(p),
            "source_arm": src_arm,
            "note": note,
        }
        if out_arm != src_arm:
            print(f"    {out_arm:<16} <- {src_arm} in {stem}"
                  + (f"  ({note})" if note else ""))
        else:
            print(f"    {out_arm:<16} <- {stem}"
                  + (f"  ({note})" if note else ""))

    blob = dict(settings)
    blob["real_scope"] = "hospital" if axis == "spec" else "pooled"
    blob["runs"] = out_runs
    blob["built_by"] = "scripts/build_mix_b200.py"
    blob["built_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob["provenance"] = provenance
    return blob


def main() -> None:
    for axis in ("spec", "gen"):
        print(f"{axis}:")
        blob = build(axis)
        dest = os.path.join(T, f"mix_{axis}.json")
        if os.path.exists(dest):
            # Keep the old pair: it is the only record of whatever produced
            # the figure that has been circulating.
            backup = dest.replace(".json", "_prebuild.json")
            if not os.path.exists(backup):
                shutil.copy2(dest, backup)
                print(f"    kept previous as {os.path.basename(backup)}")
        with open(dest, "w") as fh:
            json.dump(blob, fh, indent=1)
        print(f"  wrote {dest}  "
              + " ".join(f"{k}={blob[k]}" for k in MUST_MATCH))


if __name__ == "__main__":
    main()
