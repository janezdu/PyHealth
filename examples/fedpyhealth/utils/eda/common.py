"""Shared plumbing for the EDA analyses: tables, summary stats, output paths.

Every analysis in this package is a ``run(cfg) -> payload`` function driven by
``examples/fedpyhealth/eda.py``. What they have in common lives here so the
analyses themselves stay about their question: a printed table style, the
support-band naming test2 uses, the ICD name cache, and where files land.

Output convention
-----------------
Machine-readable artifacts (JSON/CSV) go to ``cfg["out"]`` -- ``_outputs/eda``
by default, which is gitignored. Rendered pages go to ``cfg["viz_dir"]``
(``examples/fedpyhealth/viz``), which is committed, so a page keeps working
after the run directory is cleaned. Nothing written to ``viz/`` may carry a
hospital identifier or a patient row -- see ``.llms/rules/03-data-safety.md``.
"""

import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from utils.cohort import DEFAULT_CACHE_DIR, SUPPORT_BANDS

# examples/fedpyhealth/, three levels up from utils/eda/common.py.
EXAMPLE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIZ_DIR = os.path.join(EXAMPLE_DIR, "viz")
# ICD lookups are cached here so a page can be rebuilt offline, and so a
# compute node with no outbound network still renders names.
NAME_CACHE = os.path.join(VIZ_DIR, "icd9_names.json")

#: Config keys every analysis understands. Merged under each analysis's own
#: DEFAULTS, so a top-level YAML key or a shared CLI flag reaches all of them.
SHARED_DEFAULTS = {
    "cohort_cache": DEFAULT_CACHE_DIR,
    "out": os.path.join("_outputs", "eda"),
    "viz_dir": VIZ_DIR,
    "html": True,
}


# --------------------------------------------------------------------------- #
# Printing                                                                     #
# --------------------------------------------------------------------------- #
def table(header: Sequence[str], rows: Sequence[Sequence]) -> None:
    """Right-aligned numeric table with a left-aligned first column."""
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(str(header[i])), *(len(r[i]) for r in cells)) if cells
         else len(str(header[i])) for i in range(len(header))]
    print("  " + "  ".join(str(h).ljust(w[0]) if i == 0 else str(h).rjust(w[i])
                           for i, h in enumerate(header)))
    print("  " + "-" * (sum(w) + 2 * (len(w) - 1)))
    for r in cells:
        print("  " + "  ".join(c.ljust(w[0]) if i == 0 else c.rjust(w[i])
                               for i, c in enumerate(r)))


def fmt(x: float, places: int = 2) -> str:
    """Format a float, rendering NaN as ``n/a`` rather than ``nan``."""
    return "n/a" if x != x else f"{x:.{places}f}"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #
# Stats                                                                        #
# --------------------------------------------------------------------------- #
def describe(values: np.ndarray) -> Dict[str, float]:
    """Mean and quartiles, NaN-safe on an empty input."""
    if values.size == 0:
        return {k: float("nan")
                for k in ("n", "mean", "p25", "p50", "p75", "max")}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def band_of(support: int) -> str:
    """Name the SUPPORT_BANDS bucket a code's support falls in."""
    for name, lo, hi in SUPPORT_BANDS:
        if lo <= support < hi:
            return name
    return "0"


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
def write_json(path: str, payload: dict, indent=2) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=indent)
    return path


def write_csv(path: str, rows: List[dict]) -> None:
    """Write dict rows as CSV without pulling in pandas."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_html(template: str, payload: dict, out_path: str) -> bool:
    """Inline ``payload`` into a viz template. Returns False if it is absent."""
    if not os.path.exists(template):
        print(f"  no template at {template}; skipped HTML")
        return False
    with open(template) as fh:
        html = fh.read()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html.replace('"__DATA__"', json.dumps(payload)))
    print(f"  rendered {out_path}")
    return True


# --------------------------------------------------------------------------- #
# Code names                                                                   #
# --------------------------------------------------------------------------- #
def code_names(codes: Sequence[str], enabled: bool = True,
               cache_path: str = NAME_CACHE
               ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Human-readable names, cached to disk, best-effort.

    This cohort's diagnosis strings are **not** one coding system. Most are
    ICD-9-CM, but roughly 7% of the rare pool is ICD-10-CM (``I50.9``,
    ``C71.9``, ``S02.1`` ...), so a single ``InnerMap`` leaves those unnamed.
    Both maps are tried in turn and the one that resolves a code is recorded,
    because *which* system a code came from is itself a finding: an ICD-10 code
    is often a minority spelling of a condition that is well represented under
    ICD-9, which makes its rarity notational rather than clinical.

    Optional in every sense: each map downloads a table on first use, so this
    returns whatever it manages and leaves the rest unnamed rather than failing
    the plot. Cached results make every rebuild after the first offline-safe.

    Returns:
        ``(names, system)`` -- both keyed by code; ``system`` is ``"ICD9CM"``,
        ``"ICD10CM"`` or ``""`` when nothing resolved it.
    """
    cache: Dict[str, dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            raw = json.load(fh)
        # Tolerate the older {code: name} cache written before ICD-10 was
        # added, so an existing checkout does not have to re-download.
        cache = {k: (v if isinstance(v, dict) else {"name": v, "system": ""})
                 for k, v in raw.items()}

    missing = [c for c in codes if c not in cache]
    if enabled and missing:
        for system in ("ICD9CM", "ICD10CM"):
            unresolved = [c for c in missing if not cache.get(c, {}).get("name")]
            if not unresolved:
                break
            try:
                from pyhealth.medcode import InnerMap
                inner = InnerMap.load(system)
            except Exception as exc:               # noqa: BLE001
                print(f"  {system} unavailable ({type(exc).__name__}: {exc}); "
                      "those codes will show without names")
                continue
            for code in unresolved:
                try:
                    cache[code] = {"name": inner.lookup(code),
                                   "system": system}
                except Exception:
                    cache.setdefault(code, {"name": "", "system": ""})
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, indent=0, sort_keys=True)

    return ({c: cache.get(c, {}).get("name", "") for c in codes},
            {c: cache.get(c, {}).get("system", "") for c in codes})
