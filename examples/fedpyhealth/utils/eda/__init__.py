"""Cohort and run EDA, as a set of analyses driven by one script.

Each analysis is a module here exposing three things:

``DEFAULTS``   its config keys and their built-in values (documented inline)
``SUMMARY``    one line for ``eda.py --list``
``run(cfg)``   do the work: print the report, write the artifacts, return the
               payload dict

``examples/fedpyhealth/eda.py`` is the runnable front end -- it resolves config
and calls ``run``. Nothing here parses arguments, so an analysis is equally
usable from a notebook::

    from utils.eda import resolve, load
    load("lengths").run(resolve("lengths", {"mask_folds": 8}))

Config precedence (lowest to highest)
-------------------------------------
1. ``SHARED_DEFAULTS`` + the analysis's own ``DEFAULTS``
2. top-level keys of the YAML config, which reach every analysis that has that
   key (this is how one ``cohort_cache:`` or ``code_fold:`` covers all four)
3. the YAML's per-analysis section, e.g. ``lengths: {mask_folds: 8}``
4. explicit CLI flags

The same rule routes both a top-level YAML key and a CLI flag: it applies to
every analysis whose config has that key, and to no others. Unknown keys are an
error rather than a silent no-op -- a typo'd knob that quietly does nothing is
exactly how a run gets reported under settings it never used.
"""

import importlib
from typing import Dict, Sequence

from utils.eda.common import SHARED_DEFAULTS

#: Analysis names, in the order ``--all`` runs them: cheap and dependency-free
#: first, so a broken cohort cache surfaces before an hour of parquet reads.
ANALYSES = ("curves", "prevalence_curve", "lengths", "generator_distance",
            "drift")

_MODULES: Dict[str, object] = {}


def load(name: str):
    """Import an analysis module by name (lazily -- ``drift`` pulls in torch)."""
    if name not in ANALYSES:
        raise KeyError(f"unknown analysis {name!r}; choose from "
                       f"{list(ANALYSES)}")
    if name not in _MODULES:
        _MODULES[name] = importlib.import_module(f"utils.eda.{name}")
    return _MODULES[name]


def defaults(name: str) -> dict:
    """Full default config for one analysis: shared keys plus its own."""
    return {**SHARED_DEFAULTS, **load(name).DEFAULTS}


def _claimed_by(key: str) -> str:
    """Name the first analysis that understands ``key``, or "" if none does.

    Scans in :data:`ANALYSES` order and stops at the first match, so a shared
    key is resolved without importing every module (``drift`` pulls in torch).
    """
    for name in ANALYSES:
        if key in defaults(name):
            return name
    return ""


def resolve(name: str, yaml_cfg: dict = None, cli: dict = None) -> dict:
    """Layer YAML and CLI overrides onto one analysis's defaults.

    Args:
        name: Analysis name, one of :data:`ANALYSES`.
        yaml_cfg: Parsed config file -- top-level keys plus optional
            per-analysis sections keyed by analysis name.
        cli: Explicit command-line overrides, already ``None``-filtered.

    Returns:
        A complete config dict for ``run``: every key in ``defaults(name)``,
        with nothing extra.

    Raises:
        SystemExit: On a key inside an analysis section that that analysis does
            not understand.
    """
    yaml_cfg = yaml_cfg or {}
    cli = cli or {}
    cfg = defaults(name)

    for key, value in yaml_cfg.items():
        if key in ANALYSES:
            continue                      # handled as a section below
        if key in cfg:
            cfg[key] = value

    section = yaml_cfg.get(name) or {}
    if not isinstance(section, dict):
        raise SystemExit(f"config section {name!r} must be a mapping of "
                         f"key: value, got {type(section).__name__}")
    for key, value in section.items():
        if key not in cfg:
            raise SystemExit(
                f"config section {name!r} sets unknown key {key!r}; "
                f"{name} understands {sorted(cfg)}")
        cfg[key] = value

    for key, value in cli.items():
        if key in cfg:
            cfg[key] = value
    return cfg


def check_keys(yaml_cfg: dict, names: Sequence[str] = ANALYSES) -> None:
    """Fail on a top-level config key that no analysis would ever read.

    ``names`` are the analyses about to run; their keys are checked first so
    the common case needs no extra imports. A key none of them claims is still
    allowed if some other analysis would -- a config that covers all four is
    meant to work when only two are being run -- but a key nothing claims is a
    typo, and a typo'd knob that quietly does nothing is exactly how a run gets
    reported under settings it never used.
    """
    selected = set().union(*(set(defaults(n)) for n in names)) if names else set()
    for key in yaml_cfg:
        if key in ANALYSES or key == "analyses" or key in selected:
            continue
        if not _claimed_by(key):
            raise SystemExit(
                f"config key {key!r} is not understood by any analysis. "
                f"Top-level keys must be shared knobs (e.g. "
                f"{sorted(SHARED_DEFAULTS)}); analysis-specific ones go under "
                f"a section named for the analysis, e.g.\n"
                f"  {ANALYSES[0]}:\n    {key}: ...")
