"""Fidelity against utility: does prevalence accuracy predict downstream value?

Reads the two scoring tests' JSON and puts them on one pair of axes:

``x`` (fidelity)  test1_prevalence -- how closely each regime's synthetic code
                  prevalences track the real ones, per hospital.
``y`` (utility)   test2_rare_efficacy -- how well a classifier trained on that
                  synthetic data recovers rare codes in real patients (TSTR).

Nothing here recomputes a metric. It reads what the two tests wrote, aggregates
per-hospital arms to a family median, and renders the template. The point of the
pairing is that the two need not agree: a generator can match marginals and
still produce data nothing useful can be learned from, and the page exists to
show whether that is happening.

Fidelity is a single run per arm, so there are no error bars on x. Whether y has
them depends on the test2 run: a sampled-code run with several seeds has a
spread, an all-codes run is one number. The payload says which, and the template
only draws bars when they exist rather than implying precision it does not have.
"""

import json
import os
import statistics as st
from typing import Dict, List

from utils.eda.common import VIZ_DIR, banner, fmt, render_html, table, write_json

SUMMARY = "prevalence fidelity (test1) against rare-code TSTR utility (test2)"

# Every source is named EXPLICITLY, never by the bare default that
# test1_prevalence.py / test2_rare_efficacy.py write when given no --out. Those
# two default names are cohort-agnostic, so a scoring run on any other cohort
# claims them -- an lo8_random run silently replaced both, and regenerating this
# page would have relabelled lo8 numbers as hilo8 without raising anything.
# `cohort_name` below is asserted at load time so that cannot recur quietly.
DEFAULTS = {
    "cohort_name": "hilo8_random",
    # PRIMARY target: each site's synthetic scored against its OWN test fold
    # (--real-scope hospital), capped at 8000 with the pooled rare union.
    "test1_json":
        "_outputs/results/tests/test1_prevalence_n8000_capped_pooledrare.json",
    # ALT target: the same arms scored against the POOLED cohort-wide test fold
    # (--real-scope pooled), so the head-vs-tail panel can switch between them.
    # The two answer different questions -- "does this generator match its own
    # site" vs "does it match the cohort" -- and they rank the arms differently,
    # so showing only one would present a choice of target as a fact about the
    # generators. (Key name is historical; this is the ALT block, `test1_json`
    # is the own-fold primary.)
    "test1_own_json":
        "_outputs/results/tests/test1_prevalence_pooledreal_pooledrare.json",
    # The own-site variant: carries the per-arm `own_site` block the
    # specialist-generalist panel needs. 478 codes, 4-fold census,
    # train_budget 8000.
    "test2_json": "_outputs/results/tests/test2_ownsite_budget8000.json",
    "template": os.path.join(VIZ_DIR, "fidelity_utility_template.html"),
    "page": os.path.join(VIZ_DIR, "fidelity_utility.html"),
    # Title and subtitle, so one template can render two conditions to two
    # pages. An artifact is identified by its <title>, so a variant that kept
    # the default would be indistinguishable from the baseline in a gallery --
    # and, published to the same URL, would overwrite it.
    "title": None,
    "subtitle": None,
}

# test1 reports per hospital; test2's per-hospital families do too. Both are
# reduced with the MEDIAN rather than the mean: one hospital with a wrecked R2
# (centralized's -239) would otherwise drag a mean somewhere no hospital is.
_PREV = ("PrevVal_All_Prevalence_R2", "PrevVal_All_Prevalence_Pearson",
         "PrevVal_All_Prevalence_RMSE", "PrevVal_Rare_Prevalence_R2",
         "PrevVal_Rare_Prevalence_Pearson", "PrevVal_Rare_Prevalence_RMSE")
_KEY: Dict[str, str] = {"PrevVal_All_Prevalence_R2": "prev_all_r2",
        "PrevVal_All_Prevalence_Pearson": "prev_all_pearson",
        "PrevVal_All_Prevalence_RMSE": "prev_all_rmse",
        "PrevVal_Rare_Prevalence_R2": "prev_rare_r2",
        "PrevVal_Rare_Prevalence_Pearson": "prev_rare_pearson",
        "PrevVal_Rare_Prevalence_RMSE": "prev_rare_rmse"}

REGIME_ORDER = ("local", "fedavg_ft", "fedavg", "centralized")


def _median(vals: List[float]):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def _fidelity(t1: dict) -> Dict[str, dict]:
    """Per-regime median prevalence metrics across the cohort's hospitals."""
    out = {}
    for arm, hosp in t1.get("runs", {}).items():
        row = {}
        for m in _PREV:
            # test1 writes [value, bootstrap_std]; the std is the spread of the
            # bootstrap, not of independent runs, so it is not an error bar on
            # the point and is deliberately not carried onto the chart.
            row[_KEY[m]] = _median([v[m][0] for v in hosp.values() if m in v])
        row["n_hospitals"] = len(hosp)
        out[arm] = row
    return out


def _utility(t2: dict) -> Dict[str, dict]:
    """Per-family TSTR metrics, median over that family's classifiers."""
    fams: Dict[str, list] = {}
    for a in t2.get("arms", {}).values():
        fams.setdefault(a.get("family", "?"), []).append(a)

    out = {}
    for fam, arms in fams.items():
        # .get, not [k]: a results file written before a metric existed simply
        # lacks it, and the page must render what IS there rather than refuse to
        # build. _median already drops the Nones.
        g = lambda sect, k: _median(
            [a[sect].get(k) for a in arms if isinstance(a.get(sect), dict)])
        out[fam] = {
            "tstr_ap": g("overall", "ap_macro"),
            "tstr_auroc": g("overall", "roc_auc_macro"),
            "tstr_ap_rare": g("global_rare", "ap_macro"),
            "tstr_auroc_rare": g("global_rare", "roc_auc_macro"),
            # Two F1s because the threshold is the whole question at these
            # prevalences -- see per_code_scores in test2_rare_efficacy.py.
            # _best is an optimistic upper bound, _prev is the one to quote.
            "tstr_f1_best": g("overall", "f1_best_macro"),
            "tstr_f1_prev": g("overall", "f1_prev_macro"),
            "tstr_f1_best_rare": g("global_rare", "f1_best_macro"),
            "tstr_f1_prev_rare": g("global_rare", "f1_prev_macro"),
            # recall@k is reported at three k, because k is not a neutral
            # choice on a long tail: at k=5 a classifier must put the right rare
            # code in a very short list, at k=20 it only has to keep it in view.
            # An arm can look far better at one k than another, and picking a
            # single k would hide that.
            "tstr_recall5": _median([a.get("recall_at_5") for a in arms]),
            "tstr_recall10": _median([a.get("recall_at_10") for a in arms]),
            "tstr_recall20": _median([a.get("recall_at_20") for a in arms]),
            "n_classifiers": len(arms),
            "n_records": _median([a.get("n_train_records") for a in arms]),
            "n_degenerate": sum(1 for a in arms if a.get("degenerate")),
        }
        # Bands are per-arm dicts of dicts; median each band separately.
        band_names = sorted({b for a in arms
                             for b in (a.get("bands") or {})})
        out[fam]["bands"] = {
            b: _median([a["bands"][b]["ap_macro"] for a in arms
                        if b in (a.get("bands") or {})])
            for b in band_names}
    return out


#: Utility metrics carried per hospital as well as per arm.
_UTIL = ("tstr_ap", "tstr_auroc", "tstr_ap_rare", "tstr_auroc_rare",
         "tstr_f1_best", "tstr_f1_prev", "tstr_f1_best_rare", "tstr_f1_prev_rare",
         "tstr_recall5", "tstr_recall10", "tstr_recall20")
_SRC = {"tstr_ap": ("overall", "ap_macro"),
        "tstr_auroc": ("overall", "roc_auc_macro"),
        "tstr_ap_rare": ("global_rare", "ap_macro"),
        "tstr_auroc_rare": ("global_rare", "roc_auc_macro"),
        "tstr_f1_best": ("overall", "f1_best_macro"),
        "tstr_f1_prev": ("overall", "f1_prev_macro"),
        "tstr_f1_best_rare": ("global_rare", "f1_best_macro"),
        "tstr_f1_prev_rare": ("global_rare", "f1_prev_macro")}


def _per_hospital(t1: dict, t2: dict) -> dict:
    """``{hospital: {arm: {metric: value}}}`` -- the site-level view.

    Aggregating to a median hides the thing this experiment exists to test.
    Hospital 358 holds 194 real training records and scores recall@5 0.072 on
    them against a cohort-typical 0.35; the same site's macro AP is 0.0085,
    ABOVE four larger hospitals, because AP averages over ~478 code columns most
    of which sit near their own prevalence for every arm. One number per arm
    reports the second reading and loses the first.

    Not every arm has a per-site utility. ``fedavg`` and ``centralized`` share
    one generator, so every hospital receives the identical synthetic set and
    test2 trains ONE classifier for the whole cohort -- their utility is a
    cohort-wide constant and is emitted as such rather than copied eight times
    as if it had been measured per site.
    """
    out: Dict[str, Dict[str, dict]] = {}

    # Fidelity: test1 scores every arm against every hospital's own test fold.
    for arm, hosp in t1.get("runs", {}).items():
        for hid, mets in hosp.items():
            row = out.setdefault(hid, {}).setdefault(arm, {})
            for m, key in _KEY.items():
                if m in mets:
                    row[key] = mets[m][0]

    # Utility: only the families that genuinely have one classifier per site.
    for a in t2.get("arms", {}).values():
        hid = a.get("hospital")
        if not hid:
            continue
        fam = a.get("family", "")
        # tstr:local -> "local" so it pairs with that regime's fidelity; every
        # other per-site family (notably real_local) keeps its own name and is
        # carried as a separate series rather than being folded into a regime.
        arm = fam[len("tstr:"):] if fam.startswith("tstr:") else fam
        row = out.setdefault(hid, {}).setdefault(arm, {})
        for m in _UTIL:
            if m in _SRC:
                sect, key = _SRC[m]
                v = a.get(sect, {}).get(key)
            else:
                v = a.get(m.replace("tstr_recall", "recall_at_"))
            if v is not None:
                row[m] = v
        row["n_records"] = a.get("n_train_records")

        # The same metrics restricted to this arm's OWN site's eval rows, when
        # test2 emitted them. Prefixed rather than merged: an "own_" key is
        # unmistakably a different measurement from its pooled twin, so nothing
        # downstream can read one where it meant the other. Absent for arms with
        # no own site (fedavg, centralized, real_pooled, prior), which is how the
        # page decides which series it can draw.
        own = a.get("own_site")
        if isinstance(own, dict):
            for m in _UTIL:
                if m in _SRC:
                    sect, key = _SRC[m]
                    v = own.get(sect, {}).get(key)
                else:
                    v = own.get(m.replace("tstr_recall", "recall_at_"))
                if v is not None:
                    row["own_" + m] = v
            row["own_n_val_patients"] = own.get("n_val_patients")
    return out


def run(cfg: dict) -> dict:
    for key in ("test1_json", "test2_json"):
        if not os.path.exists(cfg[key]):
            print(f"  no {cfg[key]}; run the matching test first")
            return {}
    t1 = json.load(open(cfg["test1_json"]))
    t2 = json.load(open(cfg["test2_json"]))

    # Refuse to plot two cohorts on one page. The test scripts stamp
    # cohort_name into their output, so a source swapped for another cohort's
    # run is detectable here -- and silently mixing them would put lo8 utility
    # against hilo8 fidelity on the same axes with nothing to show it.
    want = cfg.get("cohort_name")
    if want:
        got = {cfg[k]: json.load(open(cfg[k])).get("cohort_name")
               for k in ("test1_json", "test2_json")
               if os.path.exists(cfg[k])}
        wrong = {f: c for f, c in got.items() if c != want}
        if wrong:
            raise SystemExit(
                f"cohort mismatch: expected {want!r}, but "
                + "; ".join(f"{f} holds {c!r}" for f, c in wrong.items())
                + f".\nEither point the source at a {want} run, or set "
                  "cohort_name to the cohort you mean to render.")

    fid, util = _fidelity(t1), _utility(t2)
    regimes = []
    for r in REGIME_ORDER:
        if r not in fid:
            continue
        row = {"id": r, **fid[r], **util.get(f"tstr:{r}", {})}
        regimes.append(row)
    baselines = {k: v for k, v in util.items() if not k.startswith("tstr:")}

    # The record counts are the thing most likely to be misread, so they are
    # carried into the payload rather than left in the JSON: when test2 runs
    # uncapped, the pooled arms train on 8x what the per-hospital arms get, and
    # any "arm A beats arm B" reading has to survive that first.
    meta = {
        "cohort": t2.get("cohort_name") or t1.get("cohort_name"),
        "n_hospitals": len(t2.get("hospitals") or []),
        "fold": t2.get("fold"),
        "n_scored_codes": t2.get("n_scored_codes"),
        "n_pooled_rare_codes": t2.get("n_pooled_rare_codes"),
        "n_dropped_low_support": t2.get("n_dropped_low_support"),
        "n_eval_codes": t2.get("n_eval_codes"),
        "mask_folds": t2.get("mask_folds"),
        "train_budget": t2.get("train_budget"),
        "train_budget_uncapped": bool(t2.get("train_budget_uncapped")),
        "smallest_real_train_split": t2.get("smallest_real_train_split"),
        "pooled_budget": t2.get("pooled_budget"),
        "shared_generator": t2.get("regime_shared_generator") or {},
        "record_counts": {r["id"]: r.get("n_records") for r in regimes},
    }

    banner("Fidelity vs utility")
    rows = [[r["id"], fmt(r.get("prev_all_pearson"), 3),
             fmt(r.get("prev_rare_pearson"), 3), fmt(r.get("prev_all_r2"), 2),
             fmt(r.get("tstr_ap"), 4), fmt(r.get("tstr_auroc"), 3),
             int(r.get("n_records") or 0), r.get("n_classifiers", 0)]
            for r in regimes]
    for k, v in baselines.items():
        rows.append([f"[{k}]", "-", "-", "-", fmt(v.get("tstr_ap"), 4),
                     fmt(v.get("tstr_auroc"), 3),
                     int(v.get("n_records") or 0), v.get("n_classifiers", 0)])
    table(["arm", "Pearson all", "Pearson rare", "R2 all", "AP", "AUROC",
           "records", "clf"], rows)
    if meta["train_budget_uncapped"]:
        print("  train_budget is UNCAPPED: the pooled arms train on "
              f"{max(meta['record_counts'].values() or [0])} records and the "
              "per-hospital\n  arms on far fewer, so any ranking between them "
              "is confounded by volume.")

    # Both fidelity targets, same shape, for the head-vs-tail panel.
    def _fid_block(doc):
        per = {}
        for arm, hosp in doc.get("runs", {}).items():
            for hid, mets in hosp.items():
                row = per.setdefault(hid, {}).setdefault(arm, {})
                for m, key in _KEY.items():
                    if m in mets:
                        row[key] = mets[m][0]
        return {"regimes": _fidelity(doc), "per_hospital": per,
                "real_scope": doc.get("real_scope", "hospital"),
                "rare_scope": doc.get("rare_scope", "hospital")}

    by_target = {"primary": _fid_block(t1)}
    own = cfg.get("test1_own_json")
    if own and os.path.exists(own) and os.path.abspath(own) != os.path.abspath(
            cfg["test1_json"]):
        by_target["alt"] = _fid_block(json.load(open(own)))

    per_hosp = _per_hospital(t1, t2)
    meta["per_hospital_utility_arms"] = sorted(
        {arm for h in per_hosp.values() for arm, r in h.items()
         if any(k.startswith("tstr_") for k in r)})
    meta["fidelity_targets"] = {
        k: {"real_scope": v["real_scope"], "rare_scope": v["rare_scope"]}
        for k, v in by_target.items()}
    payload = {"regimes": regimes, "baselines": baselines,
               "per_hospital": per_hosp, "fidelity_by_target": by_target,
               "meta": meta}
    write_json(os.path.join(cfg["out"], "fidelity_utility.json"), payload)
    render_html(cfg["template"], payload, cfg["page"])
    if cfg.get("title") or cfg.get("subtitle"):
        html = open(cfg["page"]).read()
        if cfg.get("title"):
            html = html.replace("<title>Fidelity vs Utility</title>",
                                f"<title>{cfg['title']}</title>", 1)
            html = html.replace("<h1>Fidelity vs Utility</h1>",
                                f"<h1>{cfg['title']}</h1>", 1)
        if cfg.get("subtitle"):
            html = html.replace(
                '<p class="eyebrow">Federated synthetic EHR · eICU · 8 hospitals</p>',
                f'<p class="eyebrow">{cfg["subtitle"]}</p>', 1)
        with open(cfg["page"], "w") as fh:
            fh.write(html)
        print(f"  retitled -> {cfg.get('title')}")
    return payload
