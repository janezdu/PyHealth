#!/usr/bin/env python3
"""Build the mechanism-sweep panel from the *_v1 sweep results.

A separate page from the fidelity/utility scatter on purpose: those four
regimes answer "which federation strategy", while these sweeps answer "what
makes best-of-K work at all". Mixing them on one axis invites reading a
dropout cell as if it were comparable to the board, and it is not -- every
number on the board was produced at dropout 0.

Usage::

    python examples/fedpyhealth/scripts/fig_mechanisms.py

Reads ``_outputs/results/runs/{control_plain,dropout_high,latent_width}_v1-*.json``
and writes ``examples/fedpyhealth/viz/mechanisms.html``, self-contained.

Only macro-averaged metrics and knob values are emitted -- nothing
patient-level, no hospital ids, no paths. Safe to commit.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.getcwd(), "_outputs", "results", "runs")
TEMPLATE = os.path.join(ROOT, "viz", "mechanisms_template.html")
OUT = os.path.join(ROOT, "viz", "mechanisms.html")


def _load(pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        with open(path) as fh:
            d = json.load(fh)
        cfg = d.get("config", {})
        out.append({
            "run": d["run_name"],
            "dropout": float(cfg.get("dropout", 0.0) or 0.0),
            "latent_dim": int(cfg.get("latent_dim", 0) or 0),
            "xm_k": int(cfg.get("xm_k", 1) or 1),
            "local_epochs": int(cfg.get("local_epochs", 0) or 0),
            "n_rounds": int(cfg.get("n_rounds", 0) or 0),
            # macro_avg_metrics values are {mean, between_hospital_std,
            # n_hospitals}, not scalars. Keep the spread: on this cohort it is
            # large (rare-prevalence Pearson carries +/-0.21 across 8 sites),
            # so a mean plotted alone would imply a precision that is not there.
            "metrics": {
                k: {"mean": v.get("mean"), "std": v.get("between_hospital_std")}
                if isinstance(v, dict) else {"mean": v, "std": None}
                for k, v in d.get("macro_avg_metrics", {}).items()
            },
        })
    return out


def main():
    control = _load("control_plain_v1-*.json")
    dropout = _load("dropout_high_v1-*.json")
    latent = _load("latent_width_v1-*.json")
    trunk = _load("trunk_e10_v1-*.json")     # may not exist yet

    if not (dropout and latent):
        sys.exit(f"no sweep results under {RESULTS} -- run the sweeps first")

    # Every run reports the same macro metric set; take it from one so the
    # selector cannot offer a metric some series lacks.
    metric_keys = sorted(dropout[0]["metrics"].keys())

    payload = {
        "control": control,
        "trunk": trunk,
        "series": [
            {"id": "dropout", "label": "Dropout x K",
             "x_key": "dropout", "x_label": "dropout",
             "group_key": "xm_k", "group_label": "K",
             "runs": dropout},
            {"id": "latent", "label": "Latent width x K",
             "x_key": "latent_dim", "x_label": "latent_dim (z width)",
             "group_key": "xm_k", "group_label": "K",
             "runs": latent},
        ],
        "metrics": metric_keys,
        "meta": {
            "n_runs": len(control) + len(dropout) + len(latent) + len(trunk),
            "has_trunk": bool(trunk),
        },
    }

    with open(TEMPLATE) as fh:
        html = fh.read()
    html = html.replace("__PAYLOAD__",
                        json.dumps(payload, separators=(",", ":")))
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT}  ({payload['meta']['n_runs']} runs, "
          f"{len(metric_keys)} metrics)")


if __name__ == "__main__":
    main()
