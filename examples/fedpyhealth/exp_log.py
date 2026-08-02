#!/usr/bin/env python3
"""Experiment log + sweeper for the fedpyhealth federated-EHR runs.

A small, deterministic state engine so launching and checking runs stay in sync:

  * Launch skills call ``record`` right after ``sbatch`` -- one append-only line
    per run in ``_outputs/experiment_log.jsonl`` (job id + run_name + config).
  * A later ``check`` polls each *unprocessed* run's SLURM state, **sweeps**
    (aggregates the metrics of) the ones that have newly finished, and marks them
    ``processed`` -- so the next ``check`` only looks at what's new. Runs that are
    still RUNNING/PENDING stay unprocessed and get re-checked next time.

Everything here is login-node-safe: an ``sacct`` query, file IO, and a no-torch
python aggregate (``sweeps/aggregate_results.py``). No training is run. Always
run from the PyHealth repo root.

Subcommands:
    record   append a launched run (--job-id + --config or --run-name)
    check    poll unprocessed runs; sweep newly-finished; mark processed
    list     print the whole log as a table

Examples::

    # right after sbatch (run_name is read from the sweep config)
    .venv/bin/python examples/fedpyhealth/exp_log.py record \\
        --job-id 19545234 --config _outputs/sweeps/fedavg_opt_v1/configs/000.yaml

    # later: what finished since last time, ranked by fidelity
    .venv/bin/python examples/fedpyhealth/exp_log.py check --sort Prevalence_R2

    .venv/bin/python examples/fedpyhealth/exp_log.py list
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

LOG = "_outputs/experiment_log.jsonl"
RESULTS_DIR = "_outputs/results"
AGG = "examples/fedpyhealth/sweeps/aggregate_results.py"

# SLURM final states. COMPLETED -> sweepable; the rest -> finished-but-no-metrics.
TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
                "DEADLINE", "PREEMPTED", "CANCELLED"}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_log() -> list:
    if not os.path.exists(LOG):
        return []
    entries = []
    with open(LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_log(entries: list) -> None:
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
    tmp = LOG + ".tmp"
    with open(tmp, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, LOG)  # atomic: a kill mid-write never corrupts the log


def _run_name_from_config(path: str) -> str:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    rn = d.get("run_name")
    if not rn:
        raise SystemExit(
            f"{path} has no 'run_name'; pass --run-name explicitly. "
            f"(Sweep configs from make_sweep.py always pin a unique run_name.)")
    return rn


def _sacct_state(job_id: str):
    """Final/current SLURM state of a job's parent row, or None if unknown."""
    try:
        out = subprocess.run(
            ["sacct", "-j", str(job_id), "-X", "--noheader", "-P",
             "--format=State"],
            capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[0].split()[0]  # "CANCELLED by 12345" -> "CANCELLED"


def cmd_record(args) -> None:
    if not args.run_name and not args.config:
        raise SystemExit("record needs --config or --run-name")
    run_name = args.run_name or _run_name_from_config(args.config)
    jid = str(args.job_id)
    entries = _read_log()
    if any(e.get("job_id") == jid for e in entries):
        print(f"job {jid} already in log; skipping")
        return
    entries.append({
        "job_id": jid,
        "run_name": run_name,
        "config": args.config,
        "results_path": os.path.join(RESULTS_DIR, f"{run_name}.json"),
        "submitted_at": _now(),
        "status": "SUBMITTED",
        "processed": False,
        "processed_at": None,
        "note": args.note,
    })
    _write_log(entries)
    print(f"recorded job {jid} -> {run_name}")


def cmd_check(args) -> None:
    entries = _read_log()
    if not entries:
        print(f"empty log ({LOG}); launch + record runs first.")
        return

    newly_done, still_running, failed = [], [], []
    for e in entries:
        if e.get("processed"):
            continue
        state = _sacct_state(e["job_id"]) or "UNKNOWN"
        e["status"] = state
        if state in TERMINAL_OK:
            e["processed"], e["processed_at"] = True, _now()
            if os.path.exists(e["results_path"]):
                newly_done.append(e)
            else:
                e["note"] = (e.get("note") or "") + " [COMPLETED but no results.json]"
                failed.append(e)
        elif state in TERMINAL_BAD:
            e["processed"], e["processed_at"] = True, _now()
            failed.append(e)
        else:  # RUNNING / PENDING / REQUEUED / UNKNOWN -> re-check next time
            still_running.append(e)
    _write_log(entries)

    print(f"=== experiment check @ {_now()} ===")
    if still_running:
        print(f"\nstill in flight ({len(still_running)}):")
        for e in still_running:
            print(f"  {e['job_id']:>10}  {e['status']:10s}  {e['run_name']}")
    if failed:
        print(f"\nfinished without clean results ({len(failed)}):")
        for e in failed:
            print(f"  {e['job_id']:>10}  {e['status']:10s}  {e['run_name']}"
                  f"{e.get('note') or ''}")
        print("  -> diagnose with the slurm-inspect skill (sacct/seff + .out log)")

    if args.all:
        sweep = [e for e in entries if os.path.exists(e["results_path"])]
        label = "all recorded"
    else:
        sweep = newly_done
        label = "newly finished"
    if not sweep:
        print(f"\nno {label} run(s) with results to sweep.")
        return
    print(f"\n=== sweep: {len(sweep)} {label} run(s) "
          f"(sorted by {args.sort}{' asc' if args.asc else ' desc'}) ===")
    files = sorted({e["results_path"] for e in sweep})
    cmd = [sys.executable, AGG, "--sort", args.sort]
    if not args.asc:
        cmd.append("--desc")
    cmd += files
    subprocess.run(cmd)


def cmd_list(args) -> None:
    entries = _read_log()
    if not entries:
        print(f"empty log ({LOG})")
        return
    w = max((len(e["run_name"]) for e in entries), default=8)
    print(f"{'job_id':>10}  {'status':10}  proc  run_name")
    print("-" * (10 + 2 + 10 + 2 + 4 + 2 + w))
    for e in entries:
        proc = "yes" if e.get("processed") else "no"
        print(f"{e['job_id']:>10}  {e.get('status', '?'):10}  {proc:4}  {e['run_name']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="append a launched run to the log")
    r.add_argument("--job-id", required=True)
    r.add_argument("--config", help="config YAML (run_name is read from it)")
    r.add_argument("--run-name", help="explicit run_name (overrides --config)")
    r.add_argument("--note", default=None)
    r.set_defaults(func=cmd_record)

    c = sub.add_parser("check",
                       help="poll unprocessed runs, sweep newly-finished, mark processed")
    c.add_argument("--sort", default="Prevalence_R2", help="macro-avg metric to rank by")
    c.add_argument("--asc", action="store_true", help="sort ascending (default: descending)")
    c.add_argument("--all", action="store_true",
                   help="re-sweep every recorded run with results, not just the new ones")
    c.set_defaults(func=cmd_check)

    l = sub.add_parser("list", help="print the whole log as a table")
    l.set_defaults(func=cmd_list)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
