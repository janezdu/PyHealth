"""Run the cohort/run EDA analyses -- one script, one config, four questions.

Each analysis answers a question that came up while reading a result, and each
lives in ``utils/eda/`` with its own reasoning written down:

``prevalence_curve``    the rare-code long tail, and which codes an evaluation
                        draw lands on (manifest only, ~1 s)
``lengths``             per-code trajectory lengths and the test2 masking
                        artifact diagnostic (reads parquet, ~1 min)
``generator_distance``  how far apart the regimes' generators are in OUTPUT
                        space, from their synthetic.json (~seconds)
``drift``               how far fine-tuning moved each hospital from the
                        federated global, in WEIGHT space (loads checkpoints,
                        ~1 min)

The first two need only the cohort cache and are what runs by default. The last
two read a training run's artifacts, so name them explicitly (or ``--all``)
once the run they describe exists.

Usage
-----
::

    python eda.py                              # prevalence_curve + lengths
    python eda.py --list                       # what exists, and what it costs
    python eda.py drift --save-dir _outputs/fedavg_ft_full_hilo8_random_save
    python eda.py --all --config configs/eda.yaml
    python eda.py lengths --data-fold test --mask-folds 8

Config
------
``--config`` takes a YAML file. Top-level keys are shared -- they reach every
analysis that has that knob -- and a section named for an analysis sets that
analysis's own. CLI flags beat both. See ``configs/eda.yaml``, which documents
every key with its default.

Everything here is CPU-only and reads a frozen cohort cache plus (for the last
two) an existing run directory; nothing trains, and nothing touches eICU.

On Delta these are minutes of CPU work, so an interactive session is better
manners than a batch job holding an idle A100 (this project's SLURM accounts
are GPU-type, and Delta refuses a no-GPU job under one)::

    srun --account=bgyw-delta-gpu --partition=gpuA100x4-interactive \
         --gpus-per-node=1 --time=00:30:00 --pty bash
    .venv/bin/python examples/fedpyhealth/eda.py --all
"""

import argparse
import os
import sys

from utils.eda import ANALYSES, check_keys, load, resolve
from utils.eda.common import banner

#: What runs when no analysis is named: the ones that need only the cohort
#: cache, so a fresh checkout with no training runs still gets a result.
DEFAULT_ANALYSES = ("prevalence_curve", "lengths")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analyses", nargs="*", metavar="ANALYSIS",
                   help=f"analyses to run, from {list(ANALYSES)} "
                        f"(default: {list(DEFAULT_ANALYSES)})")
    p.add_argument("--all", action="store_true",
                   help="run every analysis, including the two that need a "
                        "training run's artifacts")
    p.add_argument("--list", action="store_true",
                   help="describe the analyses and their config keys, run none")
    p.add_argument("--config",
                   help="YAML config, applied ON TOP of the built-in defaults "
                        "and BELOW explicit CLI flags")

    shared = p.add_argument_group("shared (any analysis with the knob)")
    shared.add_argument("--cohort-cache",
                        help="frozen cohort cache directory built by "
                             "utils/cohort.py")
    shared.add_argument("--out",
                        help="directory for CSV/JSON artifacts "
                             "(default: _outputs/eda)")
    shared.add_argument("--viz-dir",
                        help="directory holding the HTML templates and the "
                             "rendered pages (default: examples/fedpyhealth/viz)")
    shared.add_argument("--no-html", action="store_true", default=None,
                        help="skip rendering the viz pages")
    shared.add_argument("--code-fold",
                        help="which fold's support decides the scored code "
                             "list; matches test2's --fold (default: val)")
    shared.add_argument("--min-positives", type=int,
                        help="minimum --code-fold positives for a code to be "
                             "scoreable, matching test2 (default: 1)")

    g = p.add_argument_group("lengths")
    g.add_argument("--data-fold", action="append", dest="data_folds",
                   metavar="FOLD",
                   help="data fold to describe; repeatable "
                        "(default: train and val)")
    g.add_argument("--mask-folds", type=int,
                   help="mask-fold count; must match the test2 run being "
                        "explained (default: 4)")

    g = p.add_argument_group("prevalence_curve")
    g.add_argument("--n-eval", type=int,
                   help="size of the evaluation draw to highlight (default: 30)")
    g.add_argument("--seed", type=int, action="append", dest="seeds",
                   help="draw seed; repeatable to overlay several draws "
                        "(default: 0)")
    g.add_argument("--no-names", action="store_true", default=None,
                   help="skip the ICD name lookup entirely")

    g = p.add_argument_group("generator_distance")
    g.add_argument("--run", action="append", dest="runs", metavar="NAME=SAVE_DIR",
                   help="explicit run to include, repeatable, same form as "
                        "test1/test2. Given any --run, the regime/suffix "
                        "defaults are ignored -- use this for sweep "
                        "directories, whose names do not follow the regime "
                        "convention")
    g.add_argument("--suffix",
                   help="run directory suffix, appended to each regime name "
                        "(default: _full_hilo8_random_save)")
    g.add_argument("--outputs",
                   help="directory holding the <regime><suffix> run dirs "
                        "(default: _outputs)")
    g.add_argument("--real-fold",
                   help="real fold the generators were trained on "
                        "(default: train)")
    g.add_argument("--rare-only", action="store_true", default=None,
                   help="restrict the prevalence vector to pooled rare codes, "
                        "so the head codes every generator gets right stop "
                        "dominating")

    g = p.add_argument_group("curves")
    g.add_argument("--run-glob", dest="glob",
                   help="shell glob matching run directories to read tb/ from "
                        "(default: _outputs/*_save). Works on runs still in "
                        "flight -- curves just end at the last flushed point")
    g.add_argument("--min-points", type=int,
                   help="drop series with fewer than this many points "
                        "(default: 3)")

    g = p.add_argument_group("drift")
    g.add_argument("--save-dir",
                   help="run directory holding fedavg_state.pt and ft_<hid>.pt")
    g.add_argument("--top-layers", type=int,
                   help="how many layers to show in the per-layer table "
                        "(default: 12)")
    return p


def _expand(value):
    """Expand ``$VAR`` / ``${VAR}`` in every string, recursively.

    Data paths differ per machine and are credentialed, so a config has to be
    able to say ``${FEDCOHORT_CACHE}/hilo8_random`` rather than baking someone's
    scratch path into a committed file (see .llms/rules/03-data-safety.md).
    Without this the reference survives as a literal and the analysis fails on a
    path that does not exist -- or worse, silently reads the wrong directory.

    An undefined variable is left as written, which ``os.path.expandvars``
    already does; the caller's "no such file" error then still names the
    unexpanded text, which is the readable failure.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _load_yaml(path: str) -> dict:
    """Load the config YAML into a dict (empty file -> {})."""
    import yaml  # lazy: only needed when --config is used
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(
            f"config {path} must be a mapping of key: value, got "
            f"{type(cfg).__name__}.")
    return _expand(cfg)


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Only the flags actually passed, as analysis config keys.

    Every override flag defaults to ``None``, so "not None" reliably means the
    user typed it -- the same convention train.py uses.
    """
    direct = ("cohort_cache", "out", "viz_dir", "code_fold", "min_positives",
              "data_folds", "mask_folds", "n_eval", "seeds", "runs", "suffix",
              "outputs", "real_fold", "save_dir", "top_layers",
              "glob", "min_points")
    over = {k: getattr(args, k) for k in direct if getattr(args, k) is not None}
    # Negative flags, stored inverted so the config key reads positively.
    if args.no_html:
        over["html"] = False
    if args.no_names:
        over["names"] = False
    if args.rare_only:
        over["rare_only"] = True
    return over


def _describe() -> None:
    """Print each analysis, what it costs, and the keys it accepts."""
    from utils.eda import defaults
    print(__doc__.split("Usage")[0].rstrip())
    for name in ANALYSES:
        mod = load(name)
        default = "  (default)" if name in DEFAULT_ANALYSES else ""
        print(f"\n{name}{default}\n  {mod.SUMMARY}")
        for key, value in sorted(defaults(name).items()):
            print(f"    {key:16s} = {value!r}")


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.list:
        _describe()
        return

    ycfg = _load_yaml(args.config) if args.config else {}

    names = (list(ANALYSES) if args.all
             else args.analyses or ycfg.get("analyses") or
             list(DEFAULT_ANALYSES))
    unknown = [n for n in names if n not in ANALYSES]
    if unknown:
        raise SystemExit(f"unknown analysis {unknown}; choose from "
                         f"{list(ANALYSES)}")
    # Preserve ANALYSES order so a multi-analysis run is cheap-first.
    names = [n for n in ANALYSES if n in names]
    check_keys(ycfg, names)

    cli = _cli_overrides(args)
    failed = []
    for name in names:
        cfg = resolve(name, ycfg, cli)
        banner(f"{name}   {load(name).SUMMARY.split(' (')[0]}")
        try:
            load(name).run(cfg)
        except SystemExit as exc:
            # One analysis missing its inputs should not cost the others their
            # output -- report it, keep going, and exit non-zero at the end.
            print(f"\n!! {name} could not run: {exc}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} of {len(names)} analyses did not run: "
              f"{', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
