"""Score a run's synthetic patients against the real cohort.

The third of the three jobs a federated-generation experiment actually does:

    train.py     trains the generator(s) and checkpoints them   (~18h, GPU)
    generate.py  samples synthetic patients from a checkpoint    (~13m, GPU)
    eval.py      scores those patients against real data         (CPU)

Keeping them apart matters because they fail and change at completely different
rates. A metric fix, a new band, a re-score on the test fold -- none of those
touch a weight, and none of them should cost an 18-hour GPU job.

This module owns the metric suite that used to run inline at the end of
training:

- **Global**: the pooled synthetic set against the pooled real cohort (every
  hospital's train + scoring fold). Produces the privacy/attack metrics --
  NNAAR, AA_es, AA_ts, MIA_* -- which exist nowhere else in the pipeline.
- **Per-hospital**: each hospital's synthetic against that hospital's OWN real
  data, giving the same privacy suite per site plus the prevalence fidelity
  metrics (``PrevVal_All_*``, ``PrevVal_Rare_*``) shared with
  ``test1_prevalence.py``.

Both land in ``_outputs/results/runs/<run_name>.json``, which is what
``exp_log.py`` reads to build a sweep leaderboard.

    python examples/fedpyhealth/eval.py \\
        --save-dir _outputs/fedavg_full_hilo8_random_save

Note the division of labour with the standalone tests: ``test1_prevalence.py``
recomputes the prevalence half from ``synthetic.json`` and ``test2_rare_efficacy``
measures downstream ML utility. The privacy suite is this module's alone.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List

import pandas as pd

from pyhealth.metrics.generative import evaluate_synthetic_ehr
from utils.cohort import (
    DEFAULT_CACHE_DIR,
    FOLDS,
    load_clients,
    load_fold,
    load_manifest,
    load_processor,
    load_synthetic,
    load_pooled_synthetic,
    manifest_sha256,
)
from test1_prevalence import (
    EVAL_SCHEMA,
    evaluate_rare_prevalence,
    real_subset_to_records,
    synthetic_to_records,
)

# Fold the synthetic data is scored against. "val" keeps the test fold held out
# until the final numbers.
EVAL_FOLD = "val"
# Tiny-sized fallbacks so the helpers stay usable without a run config.
_DEFAULT_EVAL_CFG = {
    "sample_cap": 30,
    "lstm": {"embed_dim": 16, "hidden_dim": 16, "batch_size": 16, "epochs": 3},
    "n_bootstraps": 3,
    "n_runs": 2,
}


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def evaluate_run(train_subset, test_subset, synthetic, index_to_code,
                 metrics: str = "privacy", label: str = "global",
                 eval_cfg: dict = None):
    """Build the three frames and run evaluate_synthetic_ehr for one cohort.

    ``evaluate_synthetic_ehr`` expects long-format dataframes -- ONE ROW PER
    (patient, visit, code) event -- with columns id / time / visit_codes /
    labels. This builds those and runs the suite, so the same code scores the
    pooled cohort and each hospital against its own data.

    Returns the metric dict, or None if the cohort is too small to score (small
    hospitals can have an empty train/test slice).

    ``eval_cfg`` carries the evaluator scale knobs (``sample_cap``, ``lstm``,
    ``n_bootstraps``, ``n_runs``); when None, tiny-sized defaults are used.
    """
    cfg = {**_DEFAULT_EVAL_CFG, **(eval_cfg or {})}

    train_df = pd.DataFrame(
        real_subset_to_records(train_subset, index_to_code)).astype(EVAL_SCHEMA)
    test_df = pd.DataFrame(
        real_subset_to_records(test_subset, index_to_code)).astype(EVAL_SCHEMA)
    syn_df = pd.DataFrame(synthetic_to_records(synthetic)).astype(EVAL_SCHEMA)
    print(f"  [{label}] eval rows -- train: {len(train_df)}, "
          f"test: {len(test_df)}, synthetic: {len(syn_df)}")
    if train_df.empty or test_df.empty or syn_df.empty:
        print(f"  [{label}] skipped: empty frame (too few patients to evaluate)")
        return None
    try:
        return evaluate_synthetic_ehr(
            train_ehr=train_df,
            test_ehr=test_df,
            syn_ehr=syn_df,
            sample_size=min(cfg["sample_cap"], len(train_df), len(test_df)),
            mode="lstm",
            metrics=metrics,
            lstm_params=cfg["lstm"],
            n_bootstraps=cfg["n_bootstraps"],
            n_runs=cfg["n_runs"],
        )
    except Exception as e:  # small/degenerate cohorts can trip the metric suite
        print(f"  [{label}] eval failed: {type(e).__name__}: {e}")
        return None


def print_metrics(results: Dict[str, tuple], indent: str = "  "):
    """Pretty-print a single cohort's metric dict."""
    for name, (mean, std) in results.items():
        print(f"{indent}{name:34s} {mean:.4f} +/- {std:.4f}")


def print_client_table(per_client: Dict[str, Dict[str, tuple]]):
    """Side-by-side table: one column per hospital, one row per metric."""
    hospitals = [h for h, r in per_client.items() if r]
    if not hospitals:
        print("  (no hospital had enough data to evaluate)")
        return
    metric_names: List[str] = []
    for h in hospitals:
        for name in per_client[h]:
            if name not in metric_names:
                metric_names.append(name)
    header = ["metric"] + [f"hosp {h}" for h in hospitals]
    rows = [header]
    for name in metric_names:
        cells = [name]
        for h in hospitals:
            res = per_client[h]
            if name in res:
                mean, std = res[name]
                cells.append(f"{mean:.4f}+/-{std:.4f}")
            else:
                cells.append("-")
        rows.append(cells)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    sep = "  " + "-" * (sum(widths) + 2 * len(widths))
    for ri, row in enumerate(rows):
        line = "  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))
        print(line)
        if ri == 0:
            print(sep)


def _metrics_to_json(results: Dict[str, tuple]) -> Dict[str, dict]:
    """Turn a {name: (mean, std)} metric dict into JSON-friendly nested dicts."""
    return {name: {"mean": float(mean), "std": float(std)}
            for name, (mean, std) in results.items()}


def _macro_average(per_client: Dict[str, Dict[str, tuple]]) -> Dict[str, dict]:
    """Macro-average each metric across hospitals (unweighted mean of the
    per-hospital means; ``between_hospital_std`` = spread across hospitals).
    Mirrors how results.py collapses the per-hospital table to one number."""
    names: List[str] = []
    for res in per_client.values():
        for name in res:
            if name not in names:
                names.append(name)
    out: Dict[str, dict] = {}
    for name in names:
        vals = [res[name][0] for res in per_client.values() if name in res]
        if not vals:
            continue
        out[name] = {
            "mean": float(statistics.fmean(vals)),
            "between_hospital_std": (
                float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0),
            "n_hospitals": len(vals),
        }
    return out


def save_results_json(path: str, cfg: dict, info: Dict[str, dict],
                      num_params: int, global_results, per_client,
                      split_sha: str = None, manifest: dict = None) -> str:
    """Write one run's raw stats to ``path`` as JSON.

    Captures the full config, the partition (per-hospital train/test sizes), the
    model size, the global (pooled) metrics, every per-hospital metric, and a
    macro-averaged summary -- enough to rebuild any comparison table offline and
    to aggregate a whole sweep without re-parsing SLURM logs. Metric values are
    stored as {"mean", "std"} pairs."""
    # Promoted to the top level so a sweep leaderboard can read them cheaply.
    knob_keys = ("profile", "regime", "weighting", "local_epochs", "n_rounds",
                 "ft_epochs", "lr", "embed_dim", "n_heads", "n_layers", "n_ctx",
                 "batch_size", "num_synth", "synth_per_hospital", "metrics",
                 "cohort_cache")
    payload = {
        # Kind is stamped so a file pointed at directly still identifies itself;
        # the runs/ vs tests/ split is what keeps the two shapes from mixing.
        "kind": "run",
        "run_name": cfg["run_name"],
        "regime": cfg["regime"],
        "weighting": cfg.get("weighting", "sample"),
        "key_knobs": {k: cfg.get(k) for k in knob_keys},
        "config": cfg,
        "num_params": int(num_params),
        "partition": info,
        "global_metrics": (
            _metrics_to_json(global_results) if global_results else None),
        "per_hospital_metrics": {h: _metrics_to_json(r)
                                 for h, r in per_client.items() if r},
        "macro_avg_metrics": _macro_average(per_client),
    }
    if manifest is not None:
        payload["split"] = {
            "cohort_cache": cfg.get("cohort_cache"),
            "sha256": split_sha,
            "cohort_name": manifest.get("cohort_name"),
            "fracs": manifest.get("fracs"),
            "guaranteed_folds": manifest.get("guaranteed_folds"),
            "rare_prevalence_max": manifest.get("rare_prevalence_max"),
            "rare_min_patients": manifest.get("rare_min_patients"),
            "n_pooled_rare_codes": manifest.get("n_pooled_rare_codes"),
        }
        payload["rare_code_stats"] = {
            hid: {k: h.get(k) for k in
                  ("n_rare_codes", "min_rare_prevalence", "size_band",
                   *(f"n_{f}" for f in FOLDS))}
            for hid, h in manifest.get("per_hospital", {}).items()
        }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def evaluate_and_save(
    cfg: dict,
    manifest: dict,
    clients: Dict[str, object],
    client_tests: Dict[str, object],
    pooled_train,
    pooled_test,
    synthetic: List[dict],
    client_synth: Dict[str, List[dict]],
    index_to_code: Dict[int, str],
    info: Dict[str, dict],
    num_params: int = 0,
    split_sha: str = None,
    results_path: str = None,
) -> str:
    """Run the global + per-hospital suites and persist them.

    Called both inline by ``train.py`` at the end of a run and standalone by
    ``main`` against a finished run's ``synthetic.json``. Keeping one
    implementation is the point: a metric that means one thing in-run and
    another on re-score is worse than no metric.

    Returns:
        Path of the results JSON written.
    """
    eval_cfg = {
        "sample_cap": cfg.get("eval_sample_cap",
                              _DEFAULT_EVAL_CFG["sample_cap"]),
        "lstm": cfg.get("eval_lstm", _DEFAULT_EVAL_CFG["lstm"]),
        "n_bootstraps": cfg.get("eval_n_bootstraps",
                                _DEFAULT_EVAL_CFG["n_bootstraps"]),
        "n_runs": cfg.get("eval_n_runs", _DEFAULT_EVAL_CFG["n_runs"]),
    }
    metrics = cfg.get("metrics", "privacy")

    # GLOBAL: the pooled synthetic set against the pooled real cohort. Privacy
    # by default because this task is unconditional (no labels); the utility
    # group needs a matching label_fn on both frames (see
    # pyhealth/tasks/generate_ehr.py).
    print("\n=== Global metrics (aggregated model vs pooled cohort) ===")
    global_results = evaluate_run(
        pooled_train, pooled_test, synthetic, index_to_code,
        metrics=metrics, label="global", eval_cfg=eval_cfg,
    )
    if global_results:
        print("\nGlobal generative metrics (mean +/- std):")
        print_metrics(global_results)

    # PER-CLIENT: each hospital's synthetic against its own real slices. For
    # fedavg/centralized that synthetic is the single global model's output (so
    # this exposes how evenly one model serves heterogeneous, non-IID
    # hospitals); for local and fedavg_ft it is that hospital's OWN model's
    # output, so this is each personalized baseline on its home turf.
    print("\n=== Per-client metrics (each hospital's synthetic vs its own data) ===")
    rare_by_hospital = {hid: list(h["rare_codes"])
                        for hid, h in manifest["per_hospital"].items()}
    per_client: Dict[str, Dict[str, tuple]] = {}
    for hid in clients:
        res = evaluate_run(
            clients[hid], client_tests[hid], client_synth[hid], index_to_code,
            metrics=metrics, label=f"hosp {hid}", eval_cfg=eval_cfg,
        ) or {}
        # Prevalence against this hospital's OWN scoring fold, over its rare
        # codes and over the full vocabulary -- the same function
        # test1_prevalence.py runs standalone.
        res.update(evaluate_rare_prevalence(
            client_tests[hid], client_synth[hid], index_to_code,
            rare_codes=rare_by_hospital.get(hid),
            n_bootstraps=eval_cfg["n_bootstraps"], label=f"hosp {hid}",
        ))
        if res:
            per_client[hid] = res
    print("\nPer-hospital generative metrics (mean +/- std):")
    print_client_table(per_client)

    path = results_path or os.path.join(
        "_outputs", "results", "runs", f"{cfg['run_name']}.json")
    return save_results_json(path, cfg, info, num_params, global_results,
                             per_client, split_sha=split_sha, manifest=manifest)


def load_run_config(save_dir: str, run_name: str = None) -> dict:
    """Read a run's config, preferring the copy written at training start.

    ``<save_dir>/config.json`` is written before training begins, so it exists
    even for a run that died mid-way. The results JSON is the fallback for runs
    that predate it.

    Raises:
        FileNotFoundError: If neither source exists.
    """
    if run_name is None:
        base = os.path.basename(os.path.normpath(save_dir))
        run_name = base[:-5] if base.endswith("_save") else base

    direct = os.path.join(save_dir, "config.json")
    if os.path.exists(direct):
        with open(direct) as fh:
            return json.load(fh)

    legacy = os.path.join("_outputs", "results", "runs", f"{run_name}.json")
    if os.path.exists(legacy):
        with open(legacy) as fh:
            payload = json.load(fh)
        return payload.get("config", payload)

    raise FileNotFoundError(
        f"neither {direct} nor {legacy} exists, so the config this run was "
        "trained with is unknown. config.json is written at the start of "
        "train.py; a run predating it needs its results JSON to be present."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-dir", required=True,
                   help="the run's _outputs/<run_name>_save/ folder, holding "
                        "its synthetic.json")
    p.add_argument("--run-name",
                   help="overrides the run name inferred from --save-dir")
    p.add_argument("--cohort-cache",
                   help="cohort cache to score against (default: the one the "
                        "run was trained on, from its config)")
    p.add_argument("--fold", default=EVAL_FOLD, choices=list(FOLDS),
                   help=f"real fold to score against (default: {EVAL_FOLD})")
    p.add_argument("--mix", default="proportional",
                   choices=["proportional", "uniform"],
                   help="which pooled synthetic view the GLOBAL metrics use. "
                        "'proportional' matches the real cohort's hospital mix "
                        "and is the only view comparable against real pooled "
                        "data; 'uniform' weights every site equally and is a "
                        "representation view, not a fidelity metric")
    p.add_argument("--out",
                   help="results JSON path (default: "
                        "_outputs/results/runs/<run_name>.json)")
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    cfg = load_run_config(args.save_dir, args.run_name)
    cache = args.cohort_cache or cfg.get("cohort_cache") or DEFAULT_CACHE_DIR

    manifest = load_manifest(cache)
    hospitals = [str(h) for h in manifest["hospitals"]]

    clients_by_fold = load_clients(cache, folds=("train", args.fold))
    clients = {hid: f["train"] for hid, f in clients_by_fold.items()}
    client_tests = {hid: f[args.fold] for hid, f in clients_by_fold.items()}
    pooled_train = load_fold(cache, "train")
    pooled_test = load_fold(cache, args.fold)

    index_to_code = {
        v: k for k, v
        in load_processor(cache).code_vocab.items()
    }

    per_hospital = load_synthetic(args.save_dir)
    pooled = load_pooled_synthetic(args.save_dir, args.mix)
    missing = [h for h in hospitals if h not in per_hospital]
    if missing:
        raise ValueError(
            f"synthetic.json is missing hospitals {missing}; it does not match "
            f"the cohort at {cache}."
        )

    info = {hid: {"n_total": manifest["per_hospital"][hid]["n_total"],
                  "n_train": len(clients[hid]),
                  f"n_{args.fold}": len(client_tests[hid])}
            for hid in hospitals}

    # num_params is a property of the trained model, not of the scoring pass.
    # Carry the previous value forward rather than zeroing it on a re-score.
    out_path = args.out or os.path.join(
        "_outputs", "results", "runs", f"{cfg['run_name']}.json")
    num_params = 0
    if os.path.exists(out_path):
        with open(out_path) as fh:
            num_params = json.load(fh).get("num_params", 0)

    print(f"run       : {cfg.get('run_name')}")
    print(f"regime    : {cfg.get('regime')}")
    print(f"fold      : {args.fold}   pooled mix: {args.mix} ({len(pooled)})")
    print(f"metrics   : {cfg.get('metrics', 'privacy')}")

    path = evaluate_and_save(
        cfg, manifest, clients, client_tests, pooled_train, pooled_test,
        pooled, per_hospital, index_to_code, info,
        num_params=num_params,
        split_sha=manifest_sha256(manifest),
        results_path=out_path,
    )
    print(f"\nSaved results -> {path}")


if __name__ == "__main__":
    main()
