"""The rare-code long tail, and which 30 codes an evaluation draw lands on.

Renders every pooled rare code as a prevalence-ranked curve, marks the 1%
``global_rare`` threshold, and highlights the codes drawn by
``sample_eval_codes`` so a proposed evaluation subset can be inspected before
anything is trained on it.

Why look before drawing
-----------------------
Replacing test2's K=4 mask folds with a single 30-code subset cuts classifier
training by 4x, but it also means the entire rare-code result rests on those 30
codes. Uniform sampling is representative *in expectation*; a single draw is
not. A draw that happens to take mostly 1-4 positive codes measures something
much harder than one that takes the head, and the difference will look like a
result rather than like the draw it is. This plot makes the draw visible: where
each code sits in the ranking, how much support it has, and whether the subset
spans the tail or clusters in it.

Reads ``manifest.json`` only -- no parquet, no model, no GPU. Runs in about a
second. Writes ``viz/rare_code_tail.html`` (self-contained) and
``viz/rare_code_tail.json``.
"""

import os
from typing import Dict

from utils.cohort import (
    GLOBAL_RARE_PREVALENCE_MAX,
    code_prevalence,
    load_manifest,
    sample_eval_codes,
    scored_codes,
)
from utils.eda.common import (
    band_of,
    code_names,
    render_html,
    write_json,
)

NAME = "prevalence_curve"
SUMMARY = ("rare-code prevalence tail + the codes a given eval draw lands on "
           "(manifest only; ~1 s)")

DEFAULTS = {
    # which fold's support defines the scoreable pool the draw comes from,
    # matching test2
    "code_fold": "val",
    "min_positives": 1,
    # size of the evaluation draw to highlight
    "n_eval": 30,
    # draw seeds; list several to overlay and compare draws
    "seeds": [0],
    # ICD name lookup downloads a table on first use; set false to skip it
    "names": True,
}


def run(cfg: dict) -> dict:
    """Rank every pooled rare code and mark the codes each eval seed draws."""
    seeds = list(cfg["seeds"]) or [0]

    manifest = load_manifest(cfg["cohort_cache"])
    prevalence = code_prevalence(manifest)
    strict = set(manifest["global_rare_codes"])
    scored = scored_codes(manifest, cfg["code_fold"], cfg["min_positives"])
    scored_set = set(scored)
    support = manifest["pooled_rare_support"][cfg["code_fold"]]
    n_cohort = sum(manifest["per_hospital"][h]["n_total"]
                   for h in manifest["hospitals"])

    # Re-derived prevalence must reproduce the manifest's own global_rare set.
    # It is the same arithmetic cohort.py ran at build time, so a mismatch means
    # the manifest was written by a different definition than the one in use.
    recomputed = {c for c, p in prevalence.items()
                  if p <= GLOBAL_RARE_PREVALENCE_MAX}
    if recomputed != strict:
        raise SystemExit(
            f"recomputed global-rare set ({len(recomputed)}) disagrees with "
            f"the manifest's ({len(strict)}); the cache predates the current "
            "rarity definition -- rebuild the cohort."
        )

    ranked = sorted(prevalence.items(), key=lambda kv: (-kv[1], kv[0]))
    points = [
        {
            "code": code,
            "rank": i + 1,
            "prevalence": prev,
            "n_patients": round(prev * n_cohort),
            "global_rare": code in strict,
            "scoreable": code in scored_set,
            "support": int(support.get(code, 0)),
            "band": band_of(int(support.get(code, 0))),
        }
        for i, (code, prev) in enumerate(ranked)
    ]

    draws = {str(seed): sample_eval_codes(scored, cfg["n_eval"], seed)
             for seed in seeds}

    highlighted = sorted({c for p in draws.values() for c in p})
    names, systems = code_names([p["code"] for p in points],
                                enabled=bool(cfg["names"]),
                                cache_path=os.path.join(cfg["viz_dir"],
                                                        "icd9_names.json"))
    for p in points:
        p["name"] = names.get(p["code"], "")
        p["system"] = systems.get(p["code"], "")

    n_named = sum(1 for p in points if p["name"])
    n_icd10 = sum(1 for p in points if p["system"] == "ICD10CM")
    print(f"cohort {manifest['cohort_name']}: {n_cohort} patients, "
          f"{len(points)} pooled rare codes")
    print(f"  above the {GLOBAL_RARE_PREVALENCE_MAX:.0%} global-rare line: "
          f"{sum(1 for p in points if not p['global_rare'])}")
    print(f"  scoreable on {cfg['code_fold']} "
          f"(>= {cfg['min_positives']} positives): {len(scored)}")
    print(f"  names resolved: {n_named}/{len(points)}   "
          f"ICD-10 codes: {n_icd10}")
    if n_icd10:
        # An ICD-10 code here is usually a minority spelling of a condition
        # that is common under ICD-9, so its position in the tail reflects
        # coding drift rather than clinical rarity. Say so where it is seen.
        drawn10 = sorted({c for p in draws.values() for c in p
                          if systems.get(c) == "ICD10CM"})
        print(f"    this cohort mixes coding systems; {n_icd10} rare codes "
              "are ICD-10, whose rarity may be notational")
        if drawn10:
            print(f"    ICD-10 codes inside a draw: {', '.join(drawn10)}")

    rank_of = {p["code"]: p["rank"] for p in points}
    for seed, picked in draws.items():
        bands: Dict[str, int] = {}
        for c in picked:
            band = band_of(int(support.get(c, 0)))
            bands[band] = 1 + bands.get(band, 0)
        n_gr = sum(1 for c in picked if c in strict)
        ranks = sorted(rank_of[c] for c in picked)
        print(f"\ndraw seed={seed}: {len(picked)} codes, "
              f"{n_gr} globally rare")
        print(f"  rank span {ranks[0]}-{ranks[-1]} of {len(points)}   "
              f"median rank {ranks[len(ranks) // 2]}")
        print("  support bands: "
              + ", ".join(f"{k}={v}" for k, v in sorted(bands.items())))

    payload = {
        "meta": {
            # Cohort-level aggregates only. Hospital IDs are deliberately not
            # carried into the payload: the page never displays them, and the
            # committed viz/ files stay free of site identifiers.
            "cohort": manifest["cohort_name"],
            "n_hospitals": len(manifest["hospitals"]),
            "n_cohort_patients": n_cohort,
            "n_pooled_rare": len(points),
            "n_global_rare": len(strict),
            "n_scoreable": len(scored),
            "code_fold": cfg["code_fold"],
            "min_positives": cfg["min_positives"],
            "global_rare_threshold": GLOBAL_RARE_PREVALENCE_MAX,
            "rare_prevalence_max": manifest["rare_prevalence_max"],
            "n_eval": cfg["n_eval"],
            "seeds": seeds,
            "names_resolved": n_named,
            "n_icd10": n_icd10,
        },
        "points": points,
        "draws": draws,
        "highlighted": highlighted,
    }

    viz = cfg["viz_dir"]
    path = write_json(os.path.join(viz, "rare_code_tail.json"), payload,
                      indent=None)
    print(f"\n  wrote {path}")
    if cfg["html"]:
        render_html(os.path.join(viz, "rare_code_tail_template.html"), payload,
                    os.path.join(viz, "rare_code_tail.html"))
    return payload
