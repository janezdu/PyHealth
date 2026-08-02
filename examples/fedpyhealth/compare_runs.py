#!/usr/bin/env python3
"""Compare multiple federated-EHR runs by parsing their SLURM .out logs.

The example ``ehr_eicu.py`` prints (it does not write a structured results
file), so this tool reads the SLURM logs under ``_outputs/slurm/`` -- pulling
each run's config summary and its generative-metrics block -- and prints them
side by side.

``ehr_eicu.py`` currently prints a ``Per-hospital generative metrics`` table
(one metric per row, one column per hospital). This tool macro-averages each
metric across hospitals (unweighted mean of the per-hospital means) to a single
run-level value, with the +/- showing the spread *between* hospitals. Older
logs that printed a single aggregate ``Generative metrics`` block are still
parsed too.

Usage (from the repo root)::

    # compare every log in the default dir
    python examples/fedpyhealth/compare_runs.py

    # compare specific runs (files, globs, or bare job ids all work)
    python examples/fedpyhealth/compare_runs.py _outputs/slurm/fed-ehr-eicu-123.out 124 125
    python examples/fedpyhealth/compare_runs.py '_outputs/slurm/*.out'
"""

import argparse
import glob
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

DEFAULT_GLOB = "_outputs/slurm/*.out"

_FLOAT = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf)"

# Older logs print an aggregate block ("Generative metrics") with one value per
# line:  "  NNAAR                         0.5123 +/- 0.0042"  (spaces around +/-).
_AGG_HEADER_RE = re.compile(r"Generative metrics")
_METRIC_RE = re.compile(rf"^\s+(.+?)\s+{_FLOAT}\s+\+/-\s+{_FLOAT}\s*$")

# Current logs print a per-hospital table ("Per-hospital generative metrics"):
# one metric per row, one column per hospital, each cell "mean+/-std" (no
# spaces, e.g. "0.8313+/-0.0273" or "-11.8396+/-5.3598").
_PER_HOSP_HEADER_RE = re.compile(r"Per-hospital generative metrics")
_CELL_RE = re.compile(rf"{_FLOAT}\+/-{_FLOAT}")

# Config fields scraped for context (label -> compiled regex with one group).
_CONFIG_PATTERNS = {
    "run_name": re.compile(r"RUN_NAME:\s*(\S+)"),
    "local_epochs": re.compile(r"'local_epochs':\s*(\d+)"),
    "n_rounds": re.compile(r"'n_rounds':\s*(\d+)"),
    "device": re.compile(r"^device=(\S+)"),
    "samples": re.compile(r"Total samples:\s*(\d+)"),
    "code_vocab": re.compile(r"code vocab:\s*(\d+)"),
    "params": re.compile(r"Model initialized with\s*(\d+)\s*parameters"),
    "pooled_train": re.compile(r"pooled_train=(\d+)"),
    "pooled_test": re.compile(r"pooled_test=(\d+)"),
    "num_synth": re.compile(r"Generated\s*(\d+)\s*synthetic"),
}


def resolve_inputs(tokens: List[str]) -> List[str]:
    """Turn CLI tokens (files, globs, or bare job ids) into a list of log paths."""
    if not tokens:
        tokens = [DEFAULT_GLOB]
    paths: List[str] = []
    for tok in tokens:
        if os.path.isfile(tok):
            paths.append(tok)
            continue
        hits = sorted(glob.glob(tok))
        if hits:
            paths.extend(hits)
            continue
        # treat a bare token as a job id: match *-<token>.out under the default dir
        id_hits = sorted(glob.glob(os.path.join("_outputs", "slurm", f"*{tok}*.out")))
        if id_hits:
            paths.extend(id_hits)
        else:
            print(f"warning: no log matched '{tok}'")
    # de-dup, preserve order
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def label_for(path: str) -> str:
    """Short run label: the job id if the filename ends in -<id>.out, else basename."""
    base = os.path.basename(path)
    m = re.search(r"-(\d+)\.out$", base)
    return m.group(1) if m else base.removesuffix(".out")


def parse_log(path: str) -> Tuple[Dict[str, str], Dict[str, Tuple[float, float]]]:
    """Return (config, metrics) parsed from one log file.

    config:  {field: value_str}
    metrics: {metric_name: (mean, std)}  -- run-level values.

    Two log layouts are supported. The current per-hospital table is collapsed
    to one value per metric by macro-averaging across hospitals (unweighted mean
    of the per-hospital means; the std becomes the between-hospital spread). The
    older single-value aggregate block is read verbatim.
    """
    config: Dict[str, str] = {}
    metrics: Dict[str, Tuple[float, float]] = {}
    per_hosp: Dict[str, List[float]] = {}  # metric -> per-hospital means
    mode: Optional[str] = None  # None | "aggregate" | "per_hospital"
    with open(path, "r", errors="replace") as f:
        for line in f:
            for field, pat in _CONFIG_PATTERNS.items():
                if field not in config:
                    m = pat.search(line)
                    if m:
                        config[field] = m.group(1)
            if _PER_HOSP_HEADER_RE.search(line):
                mode = "per_hospital"
                continue
            if _AGG_HEADER_RE.search(line):
                mode = "aggregate"
                continue
            if mode == "aggregate":
                m = _METRIC_RE.match(line)
                if m:
                    name, mean, std = m.group(1).strip(), m.group(2), m.group(3)
                    metrics[name] = (float(mean), float(std))
            elif mode == "per_hospital":
                first = _CELL_RE.search(line)
                if first is None:
                    # Header/separator before data, or a blank line that ends the
                    # table once we have started collecting rows.
                    if per_hosp and not line.strip():
                        mode = None
                    continue
                name = line[:first.start()].strip()
                if name:
                    cells = _CELL_RE.findall(line)
                    per_hosp.setdefault(name, []).extend(
                        float(mean) for mean, _std in cells)

    # Collapse the per-hospital table to one run-level value per metric.
    for name, means in per_hosp.items():
        agg_mean = statistics.fmean(means)
        spread = statistics.pstdev(means) if len(means) > 1 else 0.0
        metrics[name] = (agg_mean, spread)
    return config, metrics


def _human(n: Optional[str]) -> str:
    """Render a param count like 1234567 as '1.23M'."""
    if n is None:
        return "-"
    try:
        v = int(n)
    except ValueError:
        return n
    if v >= 1_000_000:
        return f"{v / 1e6:.2f}M"
    if v >= 1_000:
        return f"{v / 1e3:.1f}k"
    return str(v)


def _table(header: List[str], body: List[List[str]]) -> List[str]:
    """Format a header + body into aligned, space-padded text lines."""
    all_rows = [header] + body
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header))]

    def fmt(row: List[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    sep = "-" * (sum(widths) + 2 * len(widths))
    return [fmt(header), sep] + [fmt(r) for r in body]


def render(
    runs: List[Tuple[str, Dict[str, str], Dict[str, Tuple[float, float]]]],
    show_config: bool = False,
) -> str:
    """Comparison table: one ROW per run, one COLUMN per metric.

    Config fields are always parsed (kept on hand) but only displayed when
    ``show_config`` is set (``--show-config``).
    """
    labels = [lbl for lbl, _, _ in runs]
    out: List[str] = []
    out.append(f"Comparing {len(runs)} run(s): {', '.join(labels)}")
    out.append("(values macro-averaged across hospitals; +/- = between-hospital "
               "spread)\n")

    # union of metric names, in first-seen order -> these are the columns
    metric_names: List[str] = []
    for _, _, met in runs:
        for name in met:
            if name not in metric_names:
                metric_names.append(name)

    if not metric_names:
        out.append("(no metrics block found -- runs may still be in progress or failed)")
        return "\n".join(out)

    # one row per run, one column per metric
    header = ["run"] + metric_names
    body: List[List[str]] = []
    for label, _, met in runs:
        cells = [label]
        for name in metric_names:
            if name in met:
                mean, std = met[name]
                cells.append(f"{mean:.4f}+/-{std:.4f}")
            else:
                cells.append("-")
        body.append(cells)
    out.extend(_table(header, body))

    # config kept on hand; shown only on request (rows = runs, cols = fields)
    if show_config:
        config_fields = ["local_epochs", "n_rounds", "code_vocab", "params",
                         "samples", "pooled_train", "pooled_test", "num_synth"]
        cfg_header = ["run"] + config_fields
        cfg_body: List[List[str]] = []
        for label, cfg, _ in runs:
            cells = [label]
            for field in config_fields:
                val = cfg.get(field)
                cells.append(_human(val) if field == "params" else (val or "-"))
            cfg_body.append(cells)
        out.append("")
        out.extend(_table(cfg_header, cfg_body))

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Compare federated-EHR SLURM runs.")
    ap.add_argument("inputs", nargs="*",
                    help="log files, globs, or job ids (default: %s)" % DEFAULT_GLOB)
    ap.add_argument("--show-config", action="store_true",
                    help="also print the per-run config table (device, vocab, params, ...)")
    args = ap.parse_args()

    paths = resolve_inputs(args.inputs)
    if not paths:
        print(f"No logs found. Looked under '{DEFAULT_GLOB}'.")
        return

    runs = []
    for p in paths:
        cfg, met = parse_log(p)
        # Prefer the readable RUN_NAME printed by ehr_eicu.py; fall back to the
        # job id from the filename for older logs that predate it.
        label = cfg.get("run_name") or label_for(p)
        runs.append((label, cfg, met))

    print(render(runs, show_config=args.show_config))


if __name__ == "__main__":
    main()
