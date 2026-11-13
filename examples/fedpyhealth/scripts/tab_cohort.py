"""Paper table: what the eight clients actually look like.

Reads a frozen cohort cache and emits a LaTeX table of the per-client
descriptive statistics -- size, vocabulary, rare-code count, documentation
depth, and visit structure. Nothing is trained and nothing is recomputed from
eICU; this reads the cached folds the experiments were run on, so the table and
the results describe the same data by construction.

The table exists to make one point that the size column alone does not: the
clients differ in *documentation depth* (median distinct codes per patient runs
from 1 to 8) in a way that is uncorrelated with how many patients they hold.
Size heterogeneity is causal -- a small hospital really does have few patients.
Depth heterogeneity is largely spurious -- it is a property of how a site
records, not of who it treats -- and it is the axis a Dirichlet label-skew
benchmark cannot produce.

Client identifiers are NOT carried into the output. Hospitals are relabelled
``H1..Hn`` by descending patient count, matching
``fig_specialist_generalist.py`` so the two paper artifacts index the same way.
See ``.llms/rules/03-data-safety.md``.

Usage
-----
::

    .venv/bin/python examples/fedpyhealth/scripts/tab_cohort.py

CPU-only, a couple of seconds: it reads ~400 KB of cached parquet.
"""

import argparse
import csv
import json
import os

import polars as pl

# eICU's hospital.csv is hospital-level metadata -- bed category, teaching
# status, census region. It carries no patient rows, so reading it here does
# not bring PHI into a paper artifact. Clients stay relabelled H1..Hn and the
# real hospitalid is never emitted; see .llms/rules/03-data-safety.md.
BEDS = {">= 500": r"$\ge$500", "100 - 249": "100--249",
        "250 - 499": "250--499", "<100": r"$<$100", "": "---"}
REGION = {"South": "S", "Midwest": "MW", "Northeast": "NE", "West": "W",
          "": "---"}


def site_meta(eicu_root: str, hosps: list) -> dict:
    """``{hospitalid: {beds, teach, region}}`` from eICU's hospital.csv.

    Args:
        eicu_root: Directory holding ``hospital.csv`` (eICU-CRD v2.0).
        hosps: Hospital ids to keep.

    Returns:
        One dict per requested hospital; missing ids map to blanks.
    """
    want, out = set(map(str, hosps)), {}
    path = os.path.join(eicu_root, "hospital.csv")
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["hospitalid"] in want:
                out[r["hospitalid"]] = {
                    "beds": BEDS.get(r.get("numbedscategory", ""), "---"),
                    "teach": {"t": "yes", "f": "no"}.get(
                        r.get("teachingstatus", ""), "---"),
                    "region": REGION.get(r.get("region", ""), "---")}
    return {h: out.get(str(h), {"beds": "---", "teach": "---",
                                "region": "---"}) for h in hosps}

DEFAULTS = {
    "cache": os.environ.get("FEDCOHORT_CACHE_HILO8",
                            "_outputs/cache/fedcohort/hilo8_random"),
    "out": "paper/sections/tables/cohort_hilo8.tex",
    "folds": "train,val,test",
}


def stats(cache: str, folds: list, eicu_root: str = None) -> tuple:
    """``(rows, meta)`` -- one row per client, relabelled H1.. by size."""
    man = json.load(open(os.path.join(cache, "manifest.json")))
    per = man["per_hospital"]
    # Descending patient count, so H1 is the largest client. The manifest lists
    # hospitals in selection order, which is NOT guaranteed to be size order.
    hosps = sorted(man["hospitals"], key=lambda h: -per[h]["n_total"])

    sites = site_meta(eicu_root, hosps) if eicu_root else {}
    rows, frames = [], []
    for i, h in enumerate(hosps, 1):
        df = pl.concat([pl.read_parquet(os.path.join(cache, f"{h}.{f}.parquet"))
                        for f in folds])
        # One row per (patient, position) in the flattened trajectory, so a code
        # repeated across visits counts once per visit. n_unique is the
        # patient's distinct-code count, which is the "depth" the table reports.
        pat = df.group_by("patient_id").agg(
            pl.col("code").n_unique().alias("depth"),
            pl.col("visit_idx").n_unique().alias("visits"))
        rows.append({
            "tag": f"H{i}", "n": per[h]["n_total"], "n_train": per[h]["n_train"],
            "distinct": df["code"].n_unique(), "rare": per[h]["n_rare_codes"],
            "d25": pat["depth"].quantile(0.25), "d50": pat["depth"].median(),
            "d75": pat["depth"].quantile(0.75), "dmax": pat["depth"].max(),
            "vmean": pat["visits"].mean(), "vmax": pat["visits"].max(),
            **{k: sites.get(h, {}).get(k, "---")
               for k in ("beds", "teach", "region")},
        })
        frames.append(pat)

    pooled = pl.concat(frames)
    meta = {
        "cohort": man["cohort_name"], "n_clients": len(hosps),
        "n_patients": pooled.height,
        "distinct": man["cohort_distinct_codes"], "vocab": man["vocab_size"],
        "rare_max": man["rare_prevalence_max"],
        "rare_min_patients": man["rare_min_patients"],
        "bands": man.get("bands"), "per_band": man.get("per_band"),
        # The cohort does NOT hold two hospitals per band; report what it has.
        "band_counts": {b: sum(1 for h in hosps if per[h]["size_band"] == b)
                        for b in sorted({per[h]["size_band"] for h in hosps})},
        "has_sites": bool(eicu_root),
        "spread": rows[0]["n"] / rows[-1]["n"],
        "d50": pooled["depth"].median(),
        "d25": pooled["depth"].quantile(0.25),
        "d75": pooled["depth"].quantile(0.75),
        "vmean": pooled["visits"].mean(),
        "vmax": pooled["visits"].max(),
        "depth_ratio": max(r["d50"] for r in rows) / max(
            1e-9, min(r["d50"] for r in rows)),
    }
    return rows, meta


TEMPLATE = r"""%% GENERATED by examples/fedpyhealth/scripts/tab_cohort.py -- do not edit by hand.
%% Cohort %(cohort)s, %(n_clients)d clients, %(n_patients)d patients.
\begin{table}[t]
\centering
\caption{\textbf{The eight clients differ on two axes at once.} Cohort
\texttt{%(cohort_tex)s}: %(n_clients)d eICU hospitals stratified into four
size bands (%(band_desc)s), spanning %(spread).0f$\times$ in patient count.
%(site_sentence)sA code is
\emph{rare} at a client if it occurs in under %(rare_pct).0f\%% of that
client's patients (minimum %(rare_min_patients)d); the pooled figure is the
union over clients. The third block is the axis a size column cannot show:
median documentation depth ranges %(dmin).0f--%(dmax).0f distinct codes per
patient, spanning %(depth_ratio).0f$\times$ and uncorrelated with client
size---the deepest-coding client (H%(deepest)d) is the \emph{smallest}. Depth
is largely a property of how a site records rather than of whom it treats, and
is therefore spurious variation; size and case mix are causal. Label-skew
partitions of a common dataset produce the second axis only. Visits per patient
are near-degenerate everywhere (median 1 at every client), so codes per visit
tracks codes per patient and is not reported separately.}
\label{tab:cohort}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l%(site_cols)srrrrrrrrr}
\toprule
%(site_head1)s& \multicolumn{2}{c}{patients} & \multicolumn{2}{c}{codes}
& \multicolumn{3}{c}{distinct codes / patient}
& \multicolumn{2}{c}{visits / patient} \\
%(site_rules)s\cmidrule(lr){%(c1)d-%(c2)d}\cmidrule(lr){%(c3)d-%(c4)d}\cmidrule(lr){%(c5)d-%(c6)d}\cmidrule(lr){%(c7)d-%(c8)d}
client %(site_head2)s& total & train & distinct & rare & p25 & \textbf{med} & p75
& mean & max \\
\midrule
%(body)s\midrule
pooled %(site_pooled)s& %(n_patients)d & --- & %(distinct)d & %(pooled_rare)s
 & %(d25).0f & \textbf{%(d50).0f} & %(d75).0f & %(vmean).2f & %(vmax).0f \\
\bottomrule
\end{tabular}
\end{table}
"""

def render(rows: list, meta: dict, pooled_rare) -> str:
    has = meta.get("has_sites")
    site = (lambda r: f" & {r['beds']} & {r['teach']} & {r['region']}") if has \
        else (lambda r: "")
    body = "".join(
        f"{r['tag']}{site(r)} & {r['n']} & {r['n_train']} & {r['distinct']} & {r['rare']}"
        f" & {r['d25']:.0f} & \\textbf{{{r['d50']:.0f}}} & {r['d75']:.0f}"
        f" & {r['vmean']:.2f} & {r['vmax']:.0f} \\\\\n"
        for r in rows)
    deepest = max(rows, key=lambda r: r["d50"])
    off = 3 if has else 0
    labels = (meta.get("bands") or "").split(",")
    bc = meta.get("band_counts") or {}
    band_desc = ", ".join(
        f"{bc.get(i, 0)} in {labels[i].rstrip('-') + '+' if i == len(labels) - 1 else labels[i]}"
        for i in range(len(labels))) if labels and labels != [""] else "see text"
    n_t = sum(1 for r in rows if r.get("teach") == "yes")
    regions = sorted({r["region"] for r in rows if r.get("region") not in (None, "---")})
    site_sentence = (
        f"Of the {len(rows)} sites, {n_t} are teaching hospitals and "
        f"{len(rows) - n_t} are not, and they fall in "
        f"{len(regions)} census regions ({', '.join(regions)}); "
        "bed category, teaching status and region are eICU site metadata, "
        "not selection criteria. ") if has else ""
    return TEMPLATE % {
        **meta, "body": body,
        "band_desc": band_desc, "site_sentence": site_sentence,
        "site_cols": "ccc" if has else "",
        "site_head1": r"& \multicolumn{3}{c}{site} " if has else "",
        "site_head2": "& beds & teach. & reg. " if has else "",
        "site_pooled": "& --- & --- & --- " if has else "",
        "site_rules": r"\cmidrule(lr){2-4}" if has else "",
        "c1": 2 + off, "c2": 3 + off, "c3": 4 + off, "c4": 5 + off,
        "c5": 6 + off, "c6": 8 + off, "c7": 9 + off, "c8": 10 + off,
        "cohort_tex": meta["cohort"].replace("_", r"\_"),
        "pooled_rare": str(pooled_rare) if pooled_rare else "---",
        "rare_pct": meta["rare_max"] * 100,
        "dmin": min(r["d50"] for r in rows),
        "dmax": max(r["d50"] for r in rows),
        "deepest": int(deepest["tag"][1:]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=DEFAULTS["cache"],
                    help="frozen cohort cache directory (holds manifest.json)")
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--folds", default=DEFAULTS["folds"])
    ap.add_argument("--eicu-root", default=os.environ.get("EICU_ROOT"),
                    help="directory holding eICU's hospital.csv; adds the "
                         "bed/teaching/region columns. Hospital-level only, "
                         "no patient rows. Path comes from here or $EICU_ROOT "
                         "and is never hardcoded.")
    ap.add_argument("--pooled-rare", type=int, default=None,
                    help="size of the pooled rare-code union, as scored "
                         "(test1/test2 write it as n_pooled_rare_codes); the "
                         "manifest only holds the per-client counts")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.cache, "manifest.json")):
        raise SystemExit(
            f"no cohort cache at {args.cache}\n"
            "Point --cache at the frozen cache the experiments used, or set "
            "FEDCOHORT_CACHE_HILO8.")

    rows, meta = stats(args.cache, args.folds.split(","), args.eicu_root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(render(rows, meta, args.pooled_rare))

    print(f"cohort {meta['cohort']}: {meta['n_clients']} clients, "
          f"{meta['n_patients']} patients, {meta['distinct']} distinct codes")
    print(f"  size spread      {meta['spread']:.1f}x")
    print(f"  depth spread     {meta['depth_ratio']:.1f}x "
          f"(median distinct codes/patient, {min(r['d50'] for r in rows):.0f}"
          f"-{max(r['d50'] for r in rows):.0f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
