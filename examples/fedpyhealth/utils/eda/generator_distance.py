"""How far apart are the generators, measured by what they PRODUCE?

The weight-space question -- "how do the local models differ from the federated
one?" -- cannot be answered directly. ``train.py`` seeds once and then builds
each local model in a loop, so the eight ``local`` generators start from eight
*different* random initialisations. Two independently-initialised transformers
can compute the same function with completely unrelated weight vectors: permute
the attention heads and hidden units and nothing about the output changes. So
``cos(theta_local_i, theta_local_j)`` lands near zero whether the sites learned
the same thing or opposite things, and the number is uninterpretable.

The ``drift`` analysis's cosines were valid precisely because ``fedavg_ft``'s
eight models share one warm start -- their deltas live in the same basin.

This asks the same question where it *is* well posed: in function space. Every
generator already wrote 2000 synthetic patients to ``synthetic.json``, so
compare the distributions they emit.

What it reports, per hospital
-----------------------------
Each generator becomes a code-prevalence vector over the cohort vocabulary.
Then, for hospital *h*:

``local_h  vs  real_h``      did the isolated generator learn its own site?
``local_h  vs  fedavg``      how far is the federated consensus from where this
                             site would have gone alone?
``fedavg_ft_h vs fedavg``    how much did 2 fine-tuning epochs actually move
                             the *output* (weights moved <1%; did behaviour?)
``local_h  vs  local_j``     do isolated sites agree with each other, or is the
                             cohort genuinely heterogeneous?

Cosine is reported alongside Jensen-Shannon distance. Cosine on a prevalence
vector is dominated by the head codes both distributions share and so runs high
for almost any pair; JS is the one that moves when the tails differ. Read them
together -- a high cosine with a high JS means "same common codes, different
tail", which is exactly the regime this project cares about.

Reads ``synthetic.json`` per regime plus the cohort cache. CPU, no torch, no
GPU, a few seconds.
"""

import json
import os
from typing import Dict, Sequence

import numpy as np

from utils.cohort import load_manifest, parse_run_specs, read_trajectories
from utils.eda.common import table, write_json

NAME = "generator_distance"
SUMMARY = ("output-space distance between the regimes' generators, from their "
           "synthetic.json (a few seconds)")

REGIMES = ("local", "fedavg", "fedavg_ft", "centralized")
SHARED = ("fedavg", "centralized")   # one generator serves every hospital

DEFAULTS = {
    # explicit runs as NAME=SAVE_DIR, same form as test1/test2. Given any run,
    # the regime/suffix defaults below are ignored -- use this for sweep
    # directories, whose names do not follow the regime convention.
    "runs": [],
    # run directory suffix, appended to each regime name
    "suffix": "_full_hilo8_random_save",
    # directory holding the <regime><suffix> run dirs
    "outputs": "_outputs",
    # real fold the generators were trained on
    "real_fold": "train",
    # restrict the vector to pooled rare codes, so the head codes every
    # generator gets right stop dominating
    "rare_only": False,
}


def prevalence(patients: Sequence[dict], vocab: Dict[str, int]) -> np.ndarray:
    """Fraction of patients carrying each code, in ``vocab`` order.

    Set membership per patient, not token counts: eICU re-charts the same
    diagnosis through a stay, so counting tokens would measure charting
    verbosity rather than what the generator believes about prevalence.
    """
    v = np.zeros(len(vocab))
    for p in patients:
        seen = {c for visit in p["visits"] for c in visit}
        for c in seen:
            i = vocab.get(c)
            if i is not None:
                v[i] += 1
    return v / max(1, len(patients))


def real_prevalence(traj: Dict[str, list], vocab: Dict[str, int]) -> np.ndarray:
    """Same, from the cohort cache's ``{patient: visits}`` shape."""
    v = np.zeros(len(vocab))
    for visits in traj.values():
        for c in {c for visit in visits for c in visit}:
            i = vocab.get(c)
            if i is not None:
                v[i] += 1
    return v / max(1, len(traj))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon distance between the two vectors read as distributions.

    Normalised to sum 1 first: these are per-code prevalences, not a
    probability distribution, and without normalising a generator that simply
    emits more codes per patient would look distant from one that emits fewer
    even if the *shape* were identical.
    """
    pa, pb = a.sum(), b.sum()
    if not pa or not pb:
        return float("nan")
    p, q = a / pa, b / pb
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def run(cfg: dict) -> dict:
    """Turn each regime's synthetic.json into a prevalence vector and compare."""
    manifest = load_manifest(cfg["cohort_cache"])
    hospitals = list(manifest["hospitals"])

    codes = (sorted(manifest["pooled_rare_codes"]) if cfg["rare_only"]
             else sorted({c for h in manifest["per_hospital"]
                          for c in manifest["per_hospital"][h]["rare_codes"]}
                         | set(manifest["pooled_rare_codes"])))
    vocab = {c: i for i, c in enumerate(codes)}
    print(f"cohort {manifest['cohort_name']}   vector length {len(codes)} "
          f"({'pooled rare codes' if cfg['rare_only'] else 'rare-code union'})")

    # Explicit runs win outright. Sweep directories are named
    # ftsweep_e25_lr3e-4_save, which no regime+suffix pattern can reach.
    specs = (parse_run_specs(list(cfg["runs"])) if cfg["runs"] else
             {r: os.path.join(cfg["outputs"], f"{r}{cfg['suffix']}")
              for r in REGIMES})

    syn: Dict[str, Dict[str, np.ndarray]] = {}
    for regime, save_dir in specs.items():
        path = os.path.join(save_dir, "synthetic.json")
        if not os.path.exists(path):
            print(f"  skipping {regime}: no synthetic.json")
            continue
        with open(path) as fh:
            blob = json.load(fh)
        per = blob.get("per_hospital") or {}
        syn[regime] = {h: prevalence(per[h], vocab) for h in per}
        n = sorted({len(v) for v in per.values()})
        print(f"  {regime:12s} {len(per)} hospitals, {n} patients each")

    if not syn:
        raise SystemExit(
            f"no synthetic.json found under {os.path.abspath(cfg['outputs'])!r} "
            f"for any of {list(REGIMES)}. Point 'outputs' at the directory "
            "that holds the run dirs (it is repo-relative, so running from "
            "examples/fedpyhealth needs --outputs ../../_outputs). Without "
            "this the tables below would render empty and look like a result "
            "rather than a missing path."
        )

    traj = read_trajectories(cfg["cohort_cache"], cfg["real_fold"], hospitals)
    real = {h: real_prevalence(traj[h], vocab) for h in hospitals}

    # A shared generator emits one set for every hospital, so any of its
    # per-hospital entries is the same vector; take the first.
    def shared_vec(regime):
        d = syn.get(regime) or {}
        return next(iter(d.values())) if d else None

    fed = shared_vec("fedavg")
    cen = shared_vec("centralized")
    # With explicit run names the fedavg/centralized reference blocks below
    # simply do not apply; the per-run-vs-real table still does.

    print(f"\nPer hospital, against the {cfg['real_fold']} real distribution")
    rows = []
    for h in hospitals:
        loc = (syn.get("local") or {}).get(h)
        r = real[h]
        rows.append([
            h, manifest["per_hospital"][h]["n_train"],
            f"{cosine(loc, r):.4f}" if loc is not None else "—",
            f"{js_distance(loc, r):.4f}" if loc is not None else "—",
            f"{cosine(fed, r):.4f}" if fed is not None else "—",
            f"{js_distance(fed, r):.4f}" if fed is not None else "—",
            f"{cosine(cen, r):.4f}" if cen is not None else "—",
            f"{js_distance(cen, r):.4f}" if cen is not None else "—",
        ])
    table(["hosp", "n_train", "loc·cos", "loc·JS", "fed·cos", "fed·JS",
           "cen·cos", "cen·JS"], rows)
    print("  cos = cosine similarity (1 = identical), JS = Jensen-Shannon "
          "distance (0 = identical).\n  Lower JS is closer. Cosine runs high "
          "for any pair sharing the head codes;\n  JS is the column that moves "
          "when the tails differ.")

    if fed is not None and syn.get("local"):
        print("\nEach site's isolated generator vs the federated consensus")
        rows = []
        for h in hospitals:
            loc = syn["local"].get(h)
            ft = (syn.get("fedavg_ft") or {}).get(h)
            if loc is None:
                continue
            rows.append([
                h, manifest["per_hospital"][h]["n_train"],
                f"{cosine(loc, fed):.4f}", f"{js_distance(loc, fed):.4f}",
                f"{cosine(ft, fed):.4f}" if ft is not None else "—",
                f"{js_distance(ft, fed):.4f}" if ft is not None else "—",
            ])
        table(["hosp", "n_train", "local vs fedavg cos", "local vs fedavg JS",
               "ft vs fedavg cos", "ft vs fedavg JS"], rows)
        print("  The last two columns are the output-space version of the "
              "drift analysis's <1%\n  weight drift: if fine-tuning barely "
              "moved the weights, it should barely have\n  moved these.")

    if syn.get("local"):
        print("\nDo the isolated generators agree with each other? "
              "(pairwise JS distance)")
        hs = [h for h in hospitals if h in syn["local"]]
        head = [""] + hs
        rows = []
        off = []
        for a in hs:
            row = [a]
            for b in hs:
                if a == b:
                    row.append("—")
                    continue
                d = js_distance(syn["local"][a], syn["local"][b])
                row.append(f"{d:.3f}")
                if a < b:
                    off.append(d)
            rows.append(row)
        table(head, rows)
        if off:
            print(f"\n  mean pairwise JS {np.mean(off):.4f}   "
                  f"min {min(off):.4f}   max {max(off):.4f}")

    payload = {
        "cohort": manifest["cohort_name"],
        "vector": "pooled_rare_codes" if cfg["rare_only"] else "rare_union",
        "n_codes": len(codes),
        "real_fold": cfg["real_fold"],
        "per_hospital": {
            h: {
                "n_train": manifest["per_hospital"][h]["n_train"],
                **{f"{k}_vs_real_cos": cosine(v[h], real[h])
                   for k, v in syn.items() if h in v},
                **{f"{k}_vs_real_js": js_distance(v[h], real[h])
                   for k, v in syn.items() if h in v},
                **({"local_vs_fedavg_cos": cosine(syn["local"][h], fed),
                    "local_vs_fedavg_js": js_distance(syn["local"][h], fed)}
                   if fed is not None and h in (syn.get("local") or {}) else {}),
            }
            for h in hospitals
        },
    }
    print(f"\nwrote "
          f"{write_json(os.path.join(cfg['out'], 'generator_distance.json'), payload)}")
    return payload
