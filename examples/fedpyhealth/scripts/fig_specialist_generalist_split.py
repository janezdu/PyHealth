#!/usr/bin/env python3
"""Specialist-vs-generalist, split into two hospital subgroups side by side.

Same figure as ``fig_specialist_generalist.py`` -- each site's synthetic scored
against its OWN test fold (y, specialist) against the SAME pooled fold (x,
generalist), with the identity diagonal separating the two -- but drawn twice,
once per subgroup, on one shared pair of axes so the panels are comparable.

NOTHING IS RECOMPUTED. Both scoring passes already write per-hospital values,
so a subgroup split is a regrouping of numbers already on disk; every past
scoring run is re-splittable retroactively.

Two splits, both with known flaws (see ``--split``):

``size``      4 sites >= 1000 train patients against 4 below. The clean break,
              and the axis the cohort was constructed around.
``teaching``  3 teaching sites against 5 non-teaching, from eICU's
              ``hospital.csv``.

    python examples/fedpyhealth/scripts/fig_specialist_generalist_split.py

Sites are relabelled H1..H8 by descending training size, as in the unsplit
figure. NOTE that the teaching split necessarily reveals which relabelled sites
carry that attribute, which the unsplit figure does not -- so this writes to the
gitignored ``_outputs/figs`` rather than ``paper/figs``. Decide before any of
this goes in a paper.
"""
import argparse
import csv
import json
import os

T = "_outputs/results/tests"

#: Which scoring passes the figure offers. Each is a (specialist, generalist)
#: pair from ONE pass -- the diagonal is only a comparison when both axes were
#: measured under the same conditions.
#:
#: All of these are the 200-bootstrap re-scores. The n=5 passes are still on
#: disk and are NOT offered: that bootstrap count is where a previous result
#: turned out to be entirely noise, and a figure that silently mixes b=5 and
#: b=200 arms invites the same mistake with extra steps.
PASSES = [
    ("baselines", "Regimes + IRM",
     f"{T}/test1_baselines_spec_b200.json", f"{T}/test1_baselines_gen_b200.json"),
    ("xm", "Best-of-K",
     f"{T}/test1_xm_spec_b200.json", f"{T}/test1_xm_gen_b200.json"),
    ("xm01", "Best-of-K vs dropout",
     f"{T}/test1_xm01_spec_b200.json", f"{T}/test1_xm01_gen_b200.json"),
    ("latent", "Latent",
     f"{T}/test1_latent_spec_b200.json", f"{T}/test1_latent_gen_b200.json"),
]

#: Colour is assigned by position within a pass, so an arm keeps one colour as
#: the subgroup toggle changes but the ramps do not have to be hand-maintained
#: as arms are added to a scoring pass.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#b0503a", "#d4886a",
        "#7a6a9b", "#2b6b7a", "#5f9ea0", "#8a8f98"]
MANIFEST = os.path.expandvars("$FEDCOHORT_CACHE/hilo8_random/manifest.json")
HOSPITAL_CSV = os.path.expandvars("$EICU_ROOT/hospital.csv")

#: ``(key, label, higher_is_better, (lo_bound, hi_bound), (axis_lo, axis_hi))``.
#: The last pair is the FIXED drawn domain, identical for every pass and both
#: panels so the diagonal sits in the same place everywhere and two passes can
#: be compared by eye. It is deliberately narrower than the data: rare Pearson
#: reaches 0.245 and R^2 reaches -285, and a domain wide enough to hold those
#: compresses everything anyone wants to read into a smudge. Points outside are
#: CLIPPED TO THE EDGE AND COUNTED, never dropped.
#: The bounds are
#: what the metric CAN take, not what these runs did: Pearson r lives in
#: [-1, 1], R^2 is capped above at 1 but unbounded below, RMSE cannot be
#: negative. Padding a domain past them prints an axis tick at a value the
#: metric cannot reach, which is a figure that lies about its own scale.
#: ``None`` means unbounded on that side.
METRICS = [
    ("PrevVal_Rare_Prevalence_Pearson", "Prevalence Pearson r, rare codes",
     True, (-1.0, 1.0), (0.5, 1.0)),
    ("PrevVal_All_Prevalence_Pearson", "Prevalence Pearson r, all codes",
     True, (-1.0, 1.0), (0.5, 1.0)),
    ("PrevVal_Rare_Prevalence_R2", "Prevalence R2, rare codes", True,
     (None, 1.0), (-1.0, 1.0)),
    ("PrevVal_All_Prevalence_R2", "Prevalence R2, all codes", True,
     (None, 1.0), (-1.0, 1.0)),
    ("PrevVal_Rare_Prevalence_RMSE", "Prevalence RMSE, rare codes", False,
     (0.0, None), (0.0, 0.1)),
    ("PrevVal_All_Prevalence_RMSE", "Prevalence RMSE, all codes", False,
     (0.0, None), (0.0, 0.1)),
]
SIZE_CUT = 1000


def load_pair(spec_path, gen_path):
    """Both scoring targets, refusing anything that is not a matched pair."""
    spec, gen = json.load(open(spec_path)), json.load(open(gen_path))
    if spec.get("real_scope") != "hospital" or gen.get("real_scope") != "pooled":
        raise SystemExit("real_scope mismatch -- arguments the wrong way round?")
    for key in ("cohort_name", "fold", "rare_scope", "synth_cap"):
        if spec.get(key) != gen.get(key):
            raise SystemExit(
                f"{key} differs between targets ({spec.get(key)!r} vs "
                f"{gen.get(key)!r}); both axes must come from one pass.")
    return spec, gen


def subgroups():
    """``{hid: {label, n_train, teaching}}`` plus the H1..H8 relabelling."""
    man = json.load(open(MANIFEST))["per_hospital"]
    teaching = {}
    with open(HOSPITAL_CSV) as fh:
        for row in csv.DictReader(fh):
            teaching[row["hospitalid"]] = row.get("teachingstatus", "")
    ordered = sorted(man, key=lambda h: -man[h]["n_train"])
    return {
        hid: {
            "label": f"H{i + 1}",
            "n_train": man[hid]["n_train"],
            "size": "large" if man[hid]["n_train"] >= SIZE_CUT else "small",
            "teaching": ("teaching" if teaching.get(hid) == "t"
                         else "non-teaching"),
        }
        for i, hid in enumerate(ordered)
    }


def build_payload():
    meta = subgroups()
    passes, points, first = [], [], None
    for pid, plabel, spec_path, gen_path in PASSES:
        if not (os.path.exists(spec_path) and os.path.exists(gen_path)):
            print(f"  skipping {pid}: scoring pass not on disk")
            continue
        spec, gen = load_pair(spec_path, gen_path)
        first = first or spec
        arms = [a for a in spec["runs"] if a in gen["runs"]]
        passes.append({"id": pid, "label": plabel, "arms": arms,
                       "n_bootstraps": spec.get("n_bootstraps")})
        for i, arm in enumerate(arms):
            for hid, info in meta.items():
                sv = spec["runs"][arm].get(hid, {})
                gv = gen["runs"][arm].get(hid, {})
                vals = {}
                for key, _, _, _, _ in METRICS:
                    if key in sv and key in gv:
                        vals[key] = {"spec": sv[key][0], "gen": gv[key][0],
                                     "spec_sd": sv[key][1], "gen_sd": gv[key][1]}
                if vals:
                    points.append({
                        "pass": pid, "arm": arm, "arm_label": arm,
                        "color": RAMP[i % len(RAMP)],
                        "site": info["label"], "n_train": info["n_train"],
                        "size": info["size"], "teaching": info["teaching"],
                        "v": vals,
                    })
    if not passes:
        raise SystemExit("no scoring passes found -- run score_cohort.sh first")
    spec = first
    return {
        "points": points,
        "passes": passes,
        "metrics": [{"key": k, "label": l, "higher": h, "bounds": b,
                     "domain": dom}
                    for k, l, h, b, dom in METRICS],
        "splits": [
            {"id": "size", "label": "Training size",
             "groups": ["large", "small"],
             "note": f"large = >= {SIZE_CUT} train patients"},
            {"id": "teaching", "label": "Teaching status",
             "groups": ["teaching", "non-teaching"],
             "note": "from eICU hospital.csv"},
        ],
        "meta": {
            "cohort": spec.get("cohort_name"), "fold": spec.get("fold"),
            "n_bootstraps": spec.get("n_bootstraps"),
            "synth_cap": spec.get("synth_cap"),
            "counts": {
                s: {g: sum(1 for m in meta.values() if m[s] == g)
                    for g in ({"large", "small"} if s == "size"
                              else {"teaching", "non-teaching"})}
                for s in ("size", "teaching")},
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_outputs/figs/specialist_split_payload.json")
    a = ap.parse_args()
    p = build_payload()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(p, open(a.out, "w"), separators=(",", ":"))
    print(f"wrote {a.out}: {len(p['points'])} points across "
          f"{len(p['passes'])} passes, {len(p['metrics'])} metrics, "
          f"counts={p['meta']['counts']}")
    for q in p["passes"]:
        print(f"    {q['id']:10} b={q['n_bootstraps']:<4} {len(q['arms'])} arms: "
              f"{', '.join(q['arms'])}")
