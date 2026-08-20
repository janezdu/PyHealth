"""Training curves from the TensorBoard event files every run writes.

A "did it finish, and where should it have stopped" view, in one page rather
than a port-forwarded TensorBoard. Reads ``<save_dir>/tb`` for every matching
run and reports, per series:

``train`` / ``val``   the two loss curves. Where they separate is the point
                      past which the model is fitting its own training data
                      rather than learning -- the sweet spot early stopping is
                      trying to find.
``best``              the epoch/round with the lowest val loss.
``stopped``           where training actually ended. ``best`` well before
                      ``stopped`` means patience was generous; ``stopped`` at
                      the budget with val still falling means it was truncated,
                      not converged.
``gpu``               utilisation and memory, if the run logged them.

Reads only what is on disk, so it works on a run that is still going -- the
curves simply end at the last flushed point.
"""

import glob
import os
from typing import Dict, List

import numpy as np

from utils.eda.common import SHARED_DEFAULTS, VIZ_DIR, banner, fmt, render_html, table, write_json

SUMMARY = "train vs val loss curves per run, with the early-stopping point"

DEFAULTS = {
    # Which run directories to read. A glob so one invocation covers a whole
    # cohort's runs, including ones still in flight.
    "glob": "_outputs/*_save",
    # Drop runs with fewer than this many points -- a job that died in its
    # first epoch has a curve but nothing to say.
    "min_points": 3,
    "template": os.path.join(VIZ_DIR, "training_curves_template.html"),
    "page": os.path.join(VIZ_DIR, "training_curves.html"),
}


def _read_tb(tb_dir: str) -> Dict[str, List[float]]:
    """``{tag: [value, ...]}`` from one run's event files, in step order."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        pts = sorted(ea.Scalars(tag), key=lambda s: s.step)
        out[tag] = [round(float(p.value), 8) for p in pts]
    return out


def _tail_drop(v: List[float], frac: float = 0.2) -> float:
    """% the curve fell over its last ``frac``. Near zero means it flattened."""
    if len(v) < 2:
        return float("nan")
    n = max(2, int(len(v) * frac))
    a, b = v[-n], v[-1]
    return 100 * (a - b) / a if a else float("nan")


def run(cfg: dict) -> dict:
    dirs = sorted(d for d in glob.glob(cfg["glob"])
                  if os.path.isdir(os.path.join(d, "tb")))
    if not dirs:
        raise SystemExit(f"no run directories with a tb/ folder match "
                         f"{cfg['glob']!r}")

    runs = {}
    for d in dirs:
        name = os.path.basename(d)[:-5] if d.endswith("_save") else os.path.basename(d)
        scal = _read_tb(os.path.join(d, "tb"))
        if not scal:
            continue
        series, run_level = {}, {}
        for tag, vals in scal.items():
            if len(vals) < cfg["min_points"]:
                continue
            kind, _, who = tag.partition("/")
            if not kind.startswith("loss"):
                continue
            phase = kind.replace("loss_", "")
            # The regimes validate differently, and conflating the two makes a
            # federated run look like it has no validation at all:
            #   local            loss_val/hospital_<id>  -- per site
            #   fedavg/fedavg_ft loss_val/global         -- ONE pooled curve,
            #                    because one global model serves every site, so
            #                    the question is cohort-wide (1,217 patients is
            #                    also a far steadier signal than any one slice)
            # Anything not named hospital_* is a run-level series, not a ninth
            # hospital.
            if who.startswith("hospital_"):
                series.setdefault(who[len("hospital_"):], {})[phase] = vals
            else:
                run_level.setdefault(who, {})[phase] = vals
        # centralized trains ONE model on pooled data, so "pooled" is its only
        # series -- there are no hospital_* tags at all. Promote the run-level
        # curves BEFORE the guard below, or the run is dropped for having no
        # per-site series when in fact it has the only series it should have.
        if not series and run_level:
            series, run_level = run_level, {}
        if not series:
            continue

        # GPU, if this run logged it (only runs started after the gpu hooks
        # landed will have these).
        gpu = {}
        for tag, vals in scal.items():
            if tag.startswith(("gpu/", "gpu_local/", "gpu_ft/")):
                gpu.setdefault(tag.split("/", 1)[1], []).extend(vals)

        # A run-level val curve (fedavg/fedavg_ft) covers every hospital, so
        # attach it to each of them rather than leaving them looking unvalidated.
        pooled_val = next((v.get("val") for k, v in run_level.items()
                           if v.get("val")), None)
        val_scope = "pooled" if pooled_val else "per_hospital"
        for s in series.values():
            if pooled_val and "val" not in s:
                s["val"] = pooled_val
                s["val_is_pooled"] = True

        for who, s in series.items():
            v = s.get("val") or []
            s["n"] = len(s.get("train") or v)
            s["tail_drop_train"] = round(_tail_drop(s.get("train") or []), 3)
            if v:
                s["best_idx"] = int(np.argmin(v))
                s["best_val"] = round(float(np.min(v)), 8)
                # Stopped well after the best => patience was generous. Stopped
                # AT the best => probably truncated by the budget, not converged.
                s["stopped_after_best"] = len(v) - 1 - s["best_idx"]
        runs[name] = {"series": series, "run_level": run_level, "gpu": gpu,
                      "val_scope": val_scope,
                      "complete": os.path.exists(os.path.join(d, "synthetic.json"))}

    banner("Training curves")
    rows = []
    for name, r in sorted(runs.items()):
        for who, s in sorted(r["series"].items()):
            rows.append([
                name[:38], who, s["n"],
                fmt(s.get("best_val", float("nan")) * 1e4, 3) if "best_val" in s else "-",
                s.get("best_idx", "-"),
                s.get("stopped_after_best", "-"),
                fmt(s.get("tail_drop_train", float("nan"))) + "%",
                "done" if r["complete"] else "running",
            ])
    table(["run", "series", "pts", "best val x1e4", "best@", "after", "train tail", ""],
          rows)
    print("  'after' = points trained past the best val. 0 means training ended "
          "at its\n  own best, i.e. it was cut off by the budget rather than by "
          "convergence.")

    # Per-hospital train sizes, so the page can colour by the high/low split
    # this cohort was built around rather than by an arbitrary palette order.
    sizes = {}
    try:
        from utils.cohort import load_manifest
        man = load_manifest(cfg["cohort_cache"])
        sizes = {h: man["per_hospital"][h]["n_train"] for h in man["hospitals"]}
    except Exception:                              # noqa: BLE001
        pass                                        # page falls back to name order

    payload = {"runs": runs, "meta": {"glob": cfg["glob"], "sizes": sizes}}
    write_json(os.path.join(cfg["out"], "training_curves.json"), payload)
    render_html(cfg["template"], payload, cfg["page"])
    return payload
