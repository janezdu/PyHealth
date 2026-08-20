"""How long is the record of a patient who carries a given rare code?

Test 2 returns below-chance AUROC on every arm but ``real_pooled`` (0.303-0.466
against a 0.500 floor). Below chance is anti-correlated, not weak: flip
``fedavg_ft``'s predictions and it becomes the second-best arm. That is a
mechanism, not noise, and this analysis measures the leading suspect.

The masking artifact
--------------------
Test 2 strips a fold's rare codes from every input and asks the model to predict
them back. The mask is applied uniformly so that no record carries a visible
"something was deleted here" tell -- but uniform *application* is not uniform
*effect*. A patient who carries five of the fold's codes loses five codes of
input; a non-carrier loses none. If carriers end up with systematically shorter
inputs, the model can learn "short record -> low risk" and invert exactly as
observed, while every individual step looks correct.

So the number to read first is the one this prints at the top: the correlation
between how many of a mask fold's codes a patient carries and how long their
input is *after* that fold is stripped, alongside the same correlation *before*
stripping. If the pre-mask correlation is positive (sicker patients have longer
records) and the post-mask one is negative, masking flipped the relationship,
and the classifier's inversion has an ordinary explanation.

Four lengths, not one
---------------------
They differ by a factor of ~6 and conflating them is the easiest way to draw the
wrong conclusion:

``n_visits``            unit stays. Cohort p50 is 1, so this is near-constant.
``n_codes_raw``         raw tokens. eICU re-charts the same diagnosis through a
                        stay, so visits average ~28 codes and run to 3601.
``n_codes_dedup``       per-visit distinct, summed. **What the model sees** --
                        ``mask_and_label`` de-duplicates within a visit.
``n_codes_postmask``    ``n_codes_dedup`` with one mask fold's codes removed.

``n_codes_dedup`` sums *per-visit* distinct counts rather than taking a
record-level set, because that is what ``mask_and_label`` produces; a code in two
visits survives twice. The two agree for single-stay patients, which is the
cohort median, and the summary reports the gap so it can be checked rather than
assumed.

Writes ``per_code_lengths.csv``, ``patient_lengths.csv``, ``summary.json`` to
``out``, and renders ``viz/cohort_eda.html`` with the data inlined.

CPU only, no torch, no model loading. About a minute, dominated by parquet
reads.
"""

import os
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from utils.cohort import (
    SUPPORT_BANDS,
    assign_folds,
    load_manifest,
    read_trajectories,
    scored_codes,
)
from utils.eda.common import (
    band_of,
    banner,
    describe,
    fmt,
    render_html,
    table,
    write_csv,
    write_json,
)

NAME = "lengths"
SUMMARY = ("per-rare-code trajectory lengths + the test2 masking-artifact "
           "diagnostic (reads parquet; ~1 min)")

DEFAULTS = {
    # data folds to describe; test2 trains on train and scores on val
    "data_folds": ["train", "val"],
    # which fold's support decides the scored code list -- matches test2's
    # --fold so both report on the same codes
    "code_fold": "val",
    # minimum code_fold positives for a code to be scoreable, matching test2
    "min_positives": 1,
    # mask-fold count; MUST match the test2 run being explained, because it
    # decides which codes each patient loses and so every post-mask length here
    "mask_folds": 4,
}

# The measures carried through the whole pipeline, in report order.
MEASURES = ("n_visits", "n_codes_raw", "n_codes_dedup", "n_codes_postmask")


# --------------------------------------------------------------------------- #
# Per-patient measures                                                         #
# --------------------------------------------------------------------------- #
def patient_measures(
    visits: Sequence[Sequence[str]], mask: Set[str],
) -> Tuple[int, int, int, int, int]:
    """Length measures for one patient, with ``mask`` stripped for the last.

    De-duplication is per visit, not per record, to mirror ``mask_and_label``
    in test2_rare_efficacy.py: a code charted in two separate stays survives
    twice, because the model sees two visits.

    Args:
        visits: The patient's trajectory as lists of code strings.
        mask: Codes removed by the mask fold under consideration.

    Returns:
        ``(n_visits, n_codes_raw, n_codes_dedup, n_codes_postmask,
        n_visits_postmask)``. A patient whose ``n_codes_postmask`` is 0 is one
        test2 drops outright -- there is nothing left to predict from.
    """
    n_raw = n_dedup = n_post = n_vis_post = 0
    for visit in visits:
        distinct = set(visit)
        kept = distinct - mask
        n_raw += len(visit)
        n_dedup += len(distinct)
        n_post += len(kept)
        if kept:
            n_vis_post += 1
    return len(visits), n_raw, n_dedup, n_post, n_vis_post


# --------------------------------------------------------------------------- #
# The fold-level computation                                                   #
# --------------------------------------------------------------------------- #
def build_fold_frame(
    traj: Dict[str, Dict[str, list]],
    codes: Sequence[str],
    mask_folds: Sequence[Sequence[str]],
) -> dict:
    """Per-patient measures under every mask fold, plus the carrier index.

    Every patient is measured once per mask fold, because the post-mask length
    depends on which fold is being stripped. With K=4 that is four passes over
    ~8k patients -- cheap, and it keeps each code paired with the exact mask its
    test2 classifier saw.

    Args:
        traj: ``{hospital: {patient: visits}}`` for one data fold.
        codes: The scored rare codes, defining the carrier matrix columns.
        mask_folds: Disjoint code lists from ``assign_folds``.

    Returns:
        Dict with ``patients`` (ids), ``hospital`` (per patient), ``carriers``
        (bool matrix, patients x codes), ``premask`` (per-patient dedup length),
        and ``postmask`` (mask_fold x patient array of post-mask lengths).
    """
    index = {c: i for i, c in enumerate(codes)}
    masks = [set(f) for f in mask_folds]

    pids: List[str] = []
    hosp: List[str] = []
    rows: List[Tuple[int, int, int]] = []          # visits, raw, dedup
    carrier_rows: List[np.ndarray] = []
    post = [[] for _ in masks]
    post_visits = [[] for _ in masks]

    for hid in sorted(traj):
        for pid in sorted(traj[hid]):
            visits = traj[hid][pid]
            n_vis, n_raw, n_dedup, _, _ = patient_measures(visits, set())
            pids.append(pid)
            hosp.append(hid)
            rows.append((n_vis, n_raw, n_dedup))

            present = {c for visit in visits for c in visit}
            flags = np.zeros(len(codes), dtype=bool)
            for c in present & index.keys():
                flags[index[c]] = True
            carrier_rows.append(flags)

            for k, mask in enumerate(masks):
                _, _, _, n_post, n_vp = patient_measures(visits, mask)
                post[k].append(n_post)
                post_visits[k].append(n_vp)

    base = np.array(rows, dtype=np.int32)
    return {
        "patients": pids,
        "hospital": np.array(hosp),
        "n_visits": base[:, 0],
        "n_codes_raw": base[:, 1],
        "n_codes_dedup": base[:, 2],
        "carriers": (np.vstack(carrier_rows) if carrier_rows
                     else np.zeros((0, len(codes)), dtype=bool)),
        "postmask": np.array(post, dtype=np.int32),
        "postmask_visits": np.array(post_visits, dtype=np.int32),
    }


def artifact_correlations(frame: dict, mask_folds: Sequence[Sequence[str]],
                          codes: Sequence[str]) -> dict:
    """The diagnostic: does masking invert the length/severity relationship?

    For each mask fold, correlate how many of that fold's codes a patient
    carries against their input length, once before masking and once after. A
    positive pre-mask correlation with a negative post-mask one means masking
    turned "carries more rare codes" into "has a shorter record", which is a
    shortcut a classifier will happily learn and which inverts its output.

    Neither sign is forced. Carrying more fold codes removes more tokens, but
    patients who carry more of anything also tend to have more of everything,
    so the two effects compete and the result is genuinely informative.
    """
    index = {c: i for i, c in enumerate(codes)}
    premask = frame["n_codes_dedup"].astype(np.float64)
    out = {"per_mask_fold": [], "pooled_pre": None, "pooled_post": None}

    pooled_carried, pooled_pre, pooled_post = [], [], []
    for k, fold_codes in enumerate(mask_folds):
        cols = [index[c] for c in fold_codes if c in index]
        carried = frame["carriers"][:, cols].sum(axis=1).astype(np.float64)
        post = frame["postmask"][k].astype(np.float64)

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        out["per_mask_fold"].append({
            "mask_fold": k,
            "n_codes": len(fold_codes),
            "corr_premask": corr(carried, premask),
            "corr_postmask": corr(carried, post),
            "mean_carried": float(np.mean(carried)),
            "n_dropped_empty_input": int((post == 0).sum()),
        })
        pooled_carried.append(carried)
        pooled_pre.append(premask)
        pooled_post.append(post)

    cat_c = np.concatenate(pooled_carried)
    if cat_c.size >= 2 and np.std(cat_c) > 0:
        out["pooled_pre"] = float(
            np.corrcoef(cat_c, np.concatenate(pooled_pre))[0, 1])
        out["pooled_post"] = float(
            np.corrcoef(cat_c, np.concatenate(pooled_post))[0, 1])
    return out


def per_code_rows(
    frame: dict, codes: Sequence[str], mask_folds: Sequence[Sequence[str]],
    support: Dict[str, int], strict: Set[str], data_fold: str,
) -> List[dict]:
    """One row per scored code: carrier lengths against the non-carrier base.

    The non-carrier baseline is recomputed per code rather than taken once per
    fold. For a rare code the two populations are near-identical in size, so the
    baseline barely moves -- but "barely" is an assumption worth not making when
    the whole point is a small systematic difference.
    """
    fold_of = {c: k for k, f in enumerate(mask_folds) for c in f}
    rows = []
    for i, code in enumerate(codes):
        k = fold_of[code]
        is_carrier = frame["carriers"][:, i]
        post = frame["postmask"][k]

        car = describe(post[is_carrier])
        non = describe(post[~is_carrier])
        row = {
            "data_fold": data_fold,
            "code": code,
            "mask_fold": k,
            "n_carriers": int(is_carrier.sum()),
            "support_scoring_fold": int(support.get(code, 0)),
            "support_band": band_of(support.get(code, 0)),
            "global_rare": code in strict,
            "carrier_postmask_mean": car["mean"],
            "carrier_postmask_p50": car["p50"],
            "noncarrier_postmask_mean": non["mean"],
            "noncarrier_postmask_p50": non["p50"],
            "delta_postmask_mean": car["mean"] - non["mean"],
            "delta_postmask_p50": car["p50"] - non["p50"],
        }
        for measure in ("n_visits", "n_codes_raw", "n_codes_dedup"):
            d = describe(frame[measure][is_carrier])
            row[f"carrier_{measure}_p50"] = d["p50"]
            row[f"carrier_{measure}_mean"] = d["mean"]
        # How much rare-code company this code's carriers keep. A code whose
        # carriers each hold 20 other rare codes is a very different prediction
        # problem from one whose carriers hold nothing else.
        row["carrier_n_rare_carried_mean"] = (
            float(np.mean(frame["carriers"][is_carrier].sum(axis=1)))
            if is_carrier.any() else float("nan"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def report_fold(data_fold: str, frame: dict, rows: List[dict],
                corr: dict, manifest: dict) -> None:
    """Print the per-fold summary, artifact diagnostic first."""
    banner(f"{data_fold.upper()}  ({len(frame['patients'])} patients)")

    print("\nMasking artifact -- does stripping a fold invert length/severity?")
    table(["mask fold", "codes", "corr pre", "corr post", "mean carried",
           "dropped"],
          [[d["mask_fold"], d["n_codes"], fmt(d["corr_premask"], 3),
            fmt(d["corr_postmask"], 3), fmt(d["mean_carried"], 2),
            d["n_dropped_empty_input"]] for d in corr["per_mask_fold"]])
    print(f"\n  pooled:  pre-mask {fmt(corr['pooled_pre'], 3)}   "
          f"post-mask {fmt(corr['pooled_post'], 3)}")
    if (corr["pooled_pre"] is not None and corr["pooled_post"] is not None
            and corr["pooled_pre"] > 0 > corr["pooled_post"]):
        print("  ^ SIGN FLIP: masking turns 'carries more rare codes' into "
              "'has a shorter record'.\n    That is a shortcut a classifier "
              "can learn, and it inverts the output.")

    print("\nLength measures (per patient)")
    table(["measure", "mean", "p25", "p50", "p75", "max"],
          [[m, fmt(describe(frame[m])["mean"]),
            fmt(describe(frame[m])["p25"]), fmt(describe(frame[m])["p50"]),
            fmt(describe(frame[m])["p75"]), fmt(describe(frame[m])["max"])]
           for m in ("n_visits", "n_codes_raw", "n_codes_dedup")])

    print("\nCarrier vs non-carrier post-mask length, by support band")
    by_band: Dict[str, List[dict]] = {}
    for r in rows:
        by_band.setdefault(r["support_band"], []).append(r)
    band_rows = []
    for name, _, _ in SUPPORT_BANDS:
        group = by_band.get(name, [])
        if not group:
            continue
        band_rows.append([
            name, len(group),
            fmt(float(np.mean([g["carrier_postmask_mean"] for g in group]))),
            fmt(float(np.mean([g["noncarrier_postmask_mean"] for g in group]))),
            fmt(float(np.mean([g["delta_postmask_mean"] for g in group]))),
            fmt(float(np.mean([g["carrier_n_rare_carried_mean"]
                               for g in group]))),
        ])
    table(["support band", "codes", "carrier len", "non-carrier len",
           "delta", "rare carried"], band_rows)

    print("\nPer hospital")
    hosp_rows = []
    for hid in manifest["hospitals"]:
        sel = frame["hospital"] == hid
        if not sel.any():
            continue
        hosp_rows.append([
            hid, int(sel.sum()),
            fmt(float(np.mean(frame["n_visits"][sel]))),
            fmt(float(np.mean(frame["n_codes_raw"][sel]))),
            fmt(float(np.mean(frame["n_codes_dedup"][sel]))),
            fmt(float(np.mean(frame["carriers"][sel].sum(axis=1)))),
        ])
    table(["hospital", "patients", "visits", "raw codes", "dedup codes",
           "rare carried"], hosp_rows)


# --------------------------------------------------------------------------- #
def run(cfg: dict) -> dict:
    """Measure trajectory lengths per data fold and per scored rare code."""
    data_folds = list(cfg["data_folds"])

    manifest = load_manifest(cfg["cohort_cache"])
    hospitals = list(manifest["hospitals"])
    strict = set(manifest.get("global_rare_codes", []))
    codes = scored_codes(manifest, cfg["code_fold"], cfg["min_positives"])
    support = manifest["pooled_rare_support"][cfg["code_fold"]]
    mask_folds = assign_folds(codes, cfg["mask_folds"])

    print(f"cohort: {manifest['cohort_name']}   hospitals: {len(hospitals)}")
    print(f"scored codes: {len(codes)} "
          f"(>= {cfg['min_positives']} {cfg['code_fold']} positives, of "
          f"{len(support)} pooled rare)")
    print(f"  globally rare among them: "
          f"{len([c for c in codes if c in strict])}")
    print(f"mask folds: {len(mask_folds)} "
          f"({', '.join(str(len(f)) for f in mask_folds)} codes)")

    all_rows: List[dict] = []
    patient_rows: List[dict] = []
    summary: Dict[str, dict] = {}

    for data_fold in data_folds:
        traj = read_trajectories(cfg["cohort_cache"], data_fold, hospitals)
        frame = build_fold_frame(traj, codes, mask_folds)

        # The manifest counted support independently at cohort-build time. An
        # exact match means trajectory reading and code counting are both
        # right; a mismatch means one of them is not, and every number below
        # would be quietly wrong.
        counted = frame["carriers"].sum(axis=0)
        expected = np.array(
            [manifest["pooled_rare_support"][data_fold].get(c, 0)
             for c in codes])
        if not np.array_equal(counted, expected):
            bad = [(codes[i], int(counted[i]), int(expected[i]))
                   for i in np.flatnonzero(counted != expected)][:5]
            raise SystemExit(
                f"carrier counts disagree with the manifest for fold "
                f"{data_fold!r} on {int((counted != expected).sum())} codes "
                f"(first few, as code/counted/manifest: {bad}). The cache and "
                "the manifest are out of sync; rebuild the cohort."
            )

        corr = artifact_correlations(frame, mask_folds, codes)
        rows = per_code_rows(frame, codes, mask_folds, support, strict,
                             data_fold)
        report_fold(data_fold, frame, rows, corr, manifest)
        all_rows.extend(rows)

        for i, pid in enumerate(frame["patients"]):
            patient_rows.append({
                "data_fold": data_fold,
                "patient_id": pid,
                "hospital": str(frame["hospital"][i]),
                "n_visits": int(frame["n_visits"][i]),
                "n_codes_raw": int(frame["n_codes_raw"][i]),
                "n_codes_dedup": int(frame["n_codes_dedup"][i]),
                "n_rare_carried": int(frame["carriers"][i].sum()),
                **{f"postmask_fold{k}": int(frame["postmask"][k][i])
                   for k in range(len(mask_folds))},
            })

        summary[data_fold] = {
            "n_patients": len(frame["patients"]),
            "correlations": corr,
            "lengths": {m: describe(frame[m])
                        for m in ("n_visits", "n_codes_raw", "n_codes_dedup")},
            "per_hospital": {
                hid: {
                    "n_patients": int((frame["hospital"] == hid).sum()),
                    "mean_codes_dedup": float(np.mean(
                        frame["n_codes_dedup"][frame["hospital"] == hid])),
                    "mean_rare_carried": float(np.mean(
                        frame["carriers"][frame["hospital"] == hid]
                        .sum(axis=1))),
                }
                for hid in hospitals
                if (frame["hospital"] == hid).any()
            },
        }

    payload = {
        "meta": {
            "cohort": manifest["cohort_name"],
            "hospitals": hospitals,
            "data_folds": data_folds,
            "code_fold": cfg["code_fold"],
            "min_positives": cfg["min_positives"],
            "mask_folds": len(mask_folds),
            "n_codes": len(codes),
            "n_global_rare": len([c for c in codes if c in strict]),
        },
        "summary": summary,
        "per_code": all_rows,
    }

    out = cfg["out"]
    write_csv(os.path.join(out, "per_code_lengths.csv"), all_rows)
    write_csv(os.path.join(out, "patient_lengths.csv"), patient_rows)
    write_json(os.path.join(out, "summary.json"), payload)
    print(f"\nwrote {len(all_rows)} code rows and {len(patient_rows)} patient "
          f"rows -> {out}")

    if cfg["html"]:
        viz = cfg["viz_dir"]
        write_json(os.path.join(viz, "cohort_eda.json"), payload, indent=None)
        render_html(os.path.join(viz, "cohort_eda_template.html"), payload,
                    os.path.join(viz, "cohort_eda.html"))
    return payload
