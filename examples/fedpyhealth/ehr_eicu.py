"""Example: federated synthetic-EHR generation on eICU with HALO + FedAvg.

eICU is multi-hospital -- every patient carries a ``hospitalid`` -- which makes it
the natural vehicle for a *federated* generative-EHR setup. This single script
runs the whole pipeline end-to-end:

1. Load eICU and apply the EHRGenerationEICU task (per-unit-stay ICD-9 sequences,
   each sample tagged with its ``hospital_id``). One fitted SampleDataset gives a
   shared code vocabulary across hospitals.
2. Partition the samples by hospital into the top-K hospitals = FedAvg clients,
   holding out a pooled real test set.
3. Train ONE HALO generator with FedAvg across the hospital clients (train each
   client locally, then sample-count-weighted average the weights each round).
4. Generate synthetic patients from the aggregated model.
5. Evaluate the synthetic data globally against the real pooled train/test
   cohorts with the generative metrics suite.
6. Evaluate the same global model per hospital (client), against each hospital's
   own real data, to show how evenly one federated model serves heterogeneous
   hospitals.

The federated helpers (``partition_by_hospital``, ``average_state_dicts``,
``run_fedavg``) are inlined below so this example is self-contained. Defaults are
sized for a quick dev/smoke run on one GPU.
"""

import argparse
import json
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import eICUDataset
from pyhealth.metrics.generative import evaluate_synthetic_ehr
from pyhealth.models import HALO
from pyhealth.tasks import EHRGenerationEICU

# --- configuration ----------------------------------------------------------
# Run-invariant settings (not scale knobs):
EICU_ROOT = "/work/hdd/bgyw/janezdu/data/eicu/eicu-crd/2.0"
MIN_VISITS = 1            # eICU patients are often single-stay; keep them
SEED = 0

# Profiles select the *scale* of a run. "smoke" is the committed quick dev run
# (tiny everything, ~minutes on one GPU). "full" is a suggested production-scale
# run -- tune to your compute budget; it is a starting point, not a validated
# config. Pick one with `--profile {smoke,full}` (default: smoke), and override
# any individual knob with its `--<knob>` flag (see _build_arg_parser).
PROFILES: Dict[str, dict] = {
    "smoke": dict(
        dev=True,                  # eICUDataset dev mode caps to ~1000 patients
        max_hospitals=3,           # number of FedAvg clients (top-K by size)
        min_hospital_samples=10,   # skip hospitals with fewer patients than this
        test_frac=0.2,             # per-hospital held-out fraction (-> pooled test)
        n_rounds=2,                # FedAvg communication rounds
        local_epochs=1,            # local training epochs per client per round
        num_synth=64,              # synthetic patients to generate
        metrics="privacy",         # "privacy" | "utility" | "all" (utility needs label_fn)
        # small HALO config for the dev subset
        embed_dim=64, n_heads=2, n_layers=2, n_ctx=20, batch_size=16, lr=1e-4,
        # downsized metric evaluator
        eval_sample_cap=30,
        eval_lstm=dict(embed_dim=16, hidden_dim=16, batch_size=16, epochs=3),
        eval_n_bootstraps=3, eval_n_runs=2,
    ),
    "full": dict(
        dev=False,                 # full eICU
        max_hospitals=50,          # eICU has ~200 hospitals; raise for real federation
        min_hospital_samples=50,   # higher bar avoids tiny noisy clients on full data
        test_frac=0.2,
        n_rounds=50,               # FedAvg needs many rounds to converge
        local_epochs=2,
        num_synth=2000,            # generate enough to match the real distribution
        metrics="privacy",         # set "all" only if you wire a label_fn (see STEP 6)
        # larger HALO config for full vocabulary / longer sequences
        embed_dim=256, n_heads=4, n_layers=4, n_ctx=50, batch_size=64, lr=1e-4,
        # heavier metric evaluator for tighter confidence intervals
        eval_sample_cap=200,
        eval_lstm=dict(embed_dim=64, hidden_dim=64, batch_size=64, epochs=10),
        eval_n_bootstraps=10, eval_n_runs=5,
    ),
}

# Helper-function default args reference the smoke profile so the partition /
# FedAvg helpers stay importable standalone with sane defaults.
_SMOKE = PROFILES["smoke"]
MAX_HOSPITALS = _SMOKE["max_hospitals"]
MIN_HOSPITAL_SAMPLES = _SMOKE["min_hospital_samples"]
TEST_FRAC = _SMOKE["test_frac"]
N_ROUNDS = _SMOKE["n_rounds"]
METRICS = _SMOKE["metrics"]


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI: pick a profile, then optionally override individual knobs."""
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--profile", choices=sorted(PROFILES), default="smoke",
                   help="run scale preset (default: smoke)")
    p.add_argument("--eicu-root", default=EICU_ROOT, help="path to eICU CRD root")
    # per-knob overrides (default None -> keep the profile's value)
    p.add_argument("--max-hospitals", type=int)
    p.add_argument("--min-hospital-samples", type=int)
    p.add_argument("--n-rounds", type=int)
    p.add_argument("--local-epochs", type=int)
    p.add_argument("--num-synth", type=int)
    p.add_argument("--metrics", choices=["privacy", "utility", "all"])
    dev = p.add_mutually_exclusive_group()
    dev.add_argument("--dev", dest="dev", action="store_true", default=None,
                     help="force dev mode (small dataset)")
    dev.add_argument("--no-dev", dest="dev", action="store_false",
                     help="force full dataset")
    p.add_argument("--resume", action="store_true",
                   help="resume FedAvg from the on-disk checkpoint if present")
    p.add_argument("--ckpt-every", type=int, default=1,
                   help="checkpoint frequency in rounds (1=every round); the "
                        "final round is always checkpointed")
    p.add_argument("--cohort-file",
                   help="JSON manifest from select_cohort.py: use exactly its "
                        "hospitals as clients instead of the top-K by size")
    p.add_argument("--tb-logdir",
                   help="TensorBoard log dir (default: <save_dir>/tb). "
                        "Logs per-hospital train-loss curves.")
    p.add_argument("--no-tb", action="store_true",
                   help="disable TensorBoard logging entirely")
    p.add_argument("--log-every-epochs", type=int, default=1,
                   help="log per-hospital loss every N local epochs (last epoch "
                        "of each round is always logged)")
    return p


def build_config(argv: List[str] = None) -> dict:
    """Resolve a config dict: profile defaults, then CLI per-knob overrides."""
    args = _build_arg_parser().parse_args(argv)
    cfg = dict(PROFILES[args.profile])
    cfg["profile"] = args.profile
    cfg["eicu_root"] = args.eicu_root
    cfg["resume"] = args.resume
    cfg["ckpt_every"] = args.ckpt_every
    cfg["cohort_file"] = args.cohort_file
    cfg["tb_logdir"] = args.tb_logdir
    cfg["no_tb"] = args.no_tb
    cfg["log_every_epochs"] = args.log_every_epochs
    for knob in ("max_hospitals", "min_hospital_samples", "n_rounds",
                 "local_epochs", "num_synth", "metrics", "dev"):
        val = getattr(args, knob)
        if val is not None:
            cfg[knob] = val
    return cfg


# ----------------------------------------------------------------------------
# Partition a fitted SampleDataset into per-hospital federated clients.
#
# The key trick: ``SampleDataset.subset(indices)`` returns a view that SHARES the
# parent's fitted processors (hence the same code vocabulary / output dims). So
# fitting ONE global dataset and carving it into per-hospital subsets makes every
# client model automatically weight-compatible -- exactly what FedAvg requires.
# ----------------------------------------------------------------------------
def partition_by_hospital(
    dataset,
    field: str = "hospital_id",
    max_clients: int = MAX_HOSPITALS,
    min_samples: int = MIN_HOSPITAL_SAMPLES,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
    cohort: List[str] = None,
) -> Tuple[Dict[str, object], Dict[str, object], object, object, Dict[str, dict]]:
    """Build federated clients + a pooled held-out test set from one dataset.

    If ``cohort`` is given (an ordered list of hospital ids, e.g. from a frozen
    manifest produced by ``select_cohort.py``), exactly those hospitals are used
    as clients **in that order** -- the size-based top-K selection is skipped.
    Consuming the cohort in manifest order is required so each hospital's
    train/test split seed (``seed + position``) matches what the manifest
    recorded. When ``cohort`` is None, the largest ``max_clients`` eligible
    hospitals are chosen (the original behaviour).

    Returns:
        clients: ``{hospital_id: train_subset}`` -- the FedAvg participants.
        client_tests: ``{hospital_id: test_subset}`` -- each hospital's held-out
            slice, for per-client evaluation of the global model.
        pooled_train: subset over the union of all client train indices.
        pooled_test: subset over the union of all client test indices (the
            shared real test set every regime is evaluated against).
        info: ``{hospital_id: {"n_total","n_train","n_test"}}`` for logging.
    """
    # map each distinct hospital id to its sample indices (one streaming pass)
    field_index: Dict[str, List[int]] = {}
    for i in range(len(dataset)):
        key = str(dataset[i].get(field, "NA"))
        field_index.setdefault(key, []).append(i)

    if cohort is not None:
        # Use exactly the frozen cohort, in its given order.
        missing = [h for h in cohort if h not in field_index]
        if missing:
            raise ValueError(
                f"Cohort hospitals not present in dataset '{field}': {missing}"
            )
        chosen = [(h, field_index[h]) for h in cohort]
    else:
        # keep the largest hospitals with at least `min_samples` patients
        eligible = [(k, v) for k, v in field_index.items() if len(v) >= min_samples]
        eligible.sort(key=lambda kv: len(kv[1]), reverse=True)
        chosen = eligible[:max_clients]
        if not chosen:
            largest = sorted((len(v) for v in field_index.values()), reverse=True)[:5]
            raise ValueError(
                f"No '{field}' group has >= {min_samples} samples "
                f"(largest groups: {largest})."
            )

    clients: Dict[str, object] = {}
    client_tests: Dict[str, object] = {}
    info: Dict[str, dict] = {}
    all_train: List[int] = []
    all_test: List[int] = []
    for client_seed, (hid, idxs) in enumerate(chosen):
        rng = np.random.default_rng(seed + client_seed)
        idx = list(idxs)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        clients[hid] = dataset.subset(train_idx)
        client_tests[hid] = dataset.subset(test_idx)
        all_train.extend(train_idx)
        all_test.extend(test_idx)
        info[hid] = {
            "n_total": len(idx),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }

    pooled_train = dataset.subset(all_train)
    pooled_test = dataset.subset(all_test)
    return clients, client_tests, pooled_train, pooled_test, info


# ----------------------------------------------------------------------------
# FedAvg over a PyHealth generator (HALO). The generator's ``train_model(train,
# val, device)`` builds a fresh optimizer each call and only checkpoints with a
# val set -- that is exactly FedAvg's local step, so we drive it directly.
# ``LOCAL_EPOCHS`` is fixed at model construction (the ``epochs=`` ctor arg).
# ----------------------------------------------------------------------------
def average_state_dicts(
    states: List[Dict[str, torch.Tensor]], weights: List[float]
) -> Dict[str, torch.Tensor]:
    """Sample-count-weighted average of model state dicts (FedAvg aggregation).

    Floating-point tensors are averaged in float64 for numerical stability;
    non-float buffers are copied from the first client unchanged.
    """
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Sum of client weights must be positive.")

    avg: Dict[str, torch.Tensor] = {}
    for key, ref in states[0].items():
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for state, w in zip(states, weights):
                acc += state[key].to(torch.float64) * (w / total)
            avg[key] = acc.to(ref.dtype)
        else:
            avg[key] = ref.clone()
    return avg


def _snapshot(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _fingerprint(sizes: Dict[str, int]) -> dict:
    """Identity of a partition, so resume refuses a mismatched checkpoint."""
    return {cid: int(n) for cid, n in sorted(sizes.items())}


def _save_ckpt(path: str, completed_rounds: int, global_state, fingerprint: dict):
    """Atomically write the FedAvg checkpoint (tmp file + rename).

    The tmp-then-rename keeps the checkpoint valid even if the job is killed
    mid-write -- ``os.replace`` is atomic on POSIX, so we never leave a
    half-written file that a later resume would choke on.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {"completed_rounds": completed_rounds,
         "global_state": global_state,
         "fingerprint": fingerprint},
        tmp,
    )
    os.replace(tmp, path)


def run_fedavg(
    model,
    clients: Dict[str, object],
    n_rounds: int = N_ROUNDS,
    device: str = "cpu",
    ckpt_path: str = None,
    ckpt_every: int = 1,
    resume: bool = False,
    writer=None,
    log_every_epochs: int = 1,
    log: Callable[[str], None] = print,
) -> object:
    """Train ``model`` with FedAvg across ``clients`` for ``n_rounds`` rounds.

    The model holds the *global* weights between rounds; ``model._epochs`` (set
    at construction) is the number of local epochs per round. Returns the model
    holding the final aggregated weights.

    Checkpointing: when ``ckpt_path`` is set, the aggregated global weights are
    written atomically every ``ckpt_every`` rounds (and always after the final
    round), tagged with the number of completed rounds and a partition
    fingerprint. Less frequent checkpoints cut I/O on long runs at the cost of
    re-doing more rounds after a kill. With ``resume=True`` an existing
    checkpoint is loaded and training continues from the next round -- so a job
    killed at the wall-clock limit loses at most one round, not the whole run.
    If the checkpoint already has >= ``n_rounds`` rounds, training is skipped and
    the model is returned with the final weights (ready for generation/eval).

    TensorBoard: when ``writer`` is given, each client's mean train loss is logged
    under ``loss_train/hospital_<id>`` every ``log_every_epochs`` epochs (and on
    each client's last local epoch), on a cumulative x-axis
    (``round * local_epochs + epoch``), plus a per-round ``loss_train/round_mean``.
    This is loss only -- cheap, already-computed scalars; generation-based metrics
    stay at end-of-run.
    """
    client_ids = list(clients.keys())
    sizes = {cid: len(clients[cid]) for cid in client_ids}
    fingerprint = _fingerprint(sizes)

    global_state = _snapshot(model)
    start_round = 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Checkpoint {ckpt_path} was written for a different partition "
                f"(client sizes differ); refusing to resume. Delete it or match "
                f"the config (profile / max_hospitals / min_hospital_samples)."
            )
        global_state = ckpt["global_state"]
        start_round = int(ckpt["completed_rounds"])
        model.load_state_dict(global_state)
        log(f"FedAvg: resumed from {ckpt_path} at round {start_round}/{n_rounds}")
    elif resume and ckpt_path:
        log(f"FedAvg: --resume set but no checkpoint at {ckpt_path}; starting fresh")

    log(f"FedAvg: {len(client_ids)} clients, sizes={sizes}, rounds={n_rounds}")
    if start_round >= n_rounds:
        log(f"FedAvg: checkpoint already has {start_round} >= {n_rounds} rounds; "
            f"skipping training")
        return model

    local_epochs = int(getattr(model, "_epochs", 1))
    for r in range(start_round, n_rounds):
        snapshots: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        round_final_losses: List[float] = []
        for cid in client_ids:
            model.load_state_dict(global_state)

            def _on_epoch_end(epoch, mean_loss, cid=cid, r=r):
                """Log this client's per-epoch loss on a cumulative x-axis."""
                if epoch + 1 == local_epochs:
                    round_final_losses.append(mean_loss)
                if writer is None:
                    return
                if (epoch + 1) % log_every_epochs == 0 or epoch + 1 == local_epochs:
                    step = r * local_epochs + epoch
                    writer.add_scalar(f"loss_train/hospital_{cid}", mean_loss, step)

            model.train_model(clients[cid], val_dataset=None, device=device,
                              on_epoch_end=_on_epoch_end)
            snapshots.append(_snapshot(model))
            weights.append(float(sizes[cid]))
            log(f"  round {r + 1}/{n_rounds}  client {cid}  (n={sizes[cid]}) done")

        if writer is not None and round_final_losses:
            writer.add_scalar("loss_train/round_mean",
                              sum(round_final_losses) / len(round_final_losses), r)

        global_state = average_state_dicts(snapshots, weights)
        model.load_state_dict(global_state)
        is_last = (r + 1) == n_rounds
        if ckpt_path and ((r + 1) % ckpt_every == 0 or is_last):
            _save_ckpt(ckpt_path, r + 1, global_state, fingerprint)
            log(f"round {r + 1}/{n_rounds} aggregated + checkpointed -> {ckpt_path}")
        else:
            log(f"round {r + 1}/{n_rounds} aggregated")

    return model


# ----------------------------------------------------------------------------
# Evaluation helpers.
#
# evaluate_synthetic_ehr expects long-format dataframes -- ONE ROW PER
# (patient, visit, code) event -- with columns id / time / visit_codes / labels.
# These helpers build those frames and run the metric suite, so we can score the
# global model once on the pooled cohort AND once per hospital (client) against
# that hospital's own real data.
# ----------------------------------------------------------------------------
EVAL_SCHEMA = {"visit_codes": str, "labels": int, "time": int, "id": str}


def real_subset_to_records(subset, index_to_code: Dict[int, str]):
    """Decode a real SampleDataset subset (index tensors) into long-format rows."""
    for sample in subset:
        pid = str(sample["patient_id"])
        for t, visit in enumerate(sample["visits"].tolist()):
            for idx in visit:
                code = index_to_code.get(int(idx))
                if code in (None, "<pad>", "<unk>"):
                    continue
                yield {"id": pid, "time": t, "visit_codes": code, "labels": 0}


def synthetic_to_records(patients: List[Dict]):
    """Convert generator output [{patient_id, visits:[[code]]}] into long-format rows."""
    for p in patients:
        pid = str(p["patient_id"])
        for t, visit in enumerate(p["visits"]):
            for code in visit:
                yield {"id": pid, "time": t, "visit_codes": str(code), "labels": 0}


def evaluate_run(train_subset, test_subset, synthetic, index_to_code,
                 metrics: str = METRICS, label: str = "global",
                 eval_cfg: dict = None):
    """Build the three frames and run evaluate_synthetic_ehr for one cohort.

    Returns the metric dict, or None if the cohort is too small to score (small
    hospitals can have an empty train/test slice). The same ``synthetic`` (from
    the single aggregated global model) is scored against each cohort.

    ``eval_cfg`` carries the metric-evaluator scale knobs (``sample_cap``,
    ``lstm``, ``n_bootstraps``, ``n_runs``); when None, smoke-sized defaults are
    used so the helper stays usable standalone.
    """
    cfg = eval_cfg or {}
    sample_cap = cfg.get("sample_cap", 30)
    lstm = cfg.get("lstm", {"embed_dim": 16, "hidden_dim": 16, "batch_size": 16, "epochs": 3})
    n_bootstraps = cfg.get("n_bootstraps", 3)
    n_runs = cfg.get("n_runs", 2)

    train_df = pd.DataFrame(real_subset_to_records(train_subset, index_to_code)).astype(EVAL_SCHEMA)
    test_df = pd.DataFrame(real_subset_to_records(test_subset, index_to_code)).astype(EVAL_SCHEMA)
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
            sample_size=min(sample_cap, len(train_df), len(test_df)),
            mode="lstm",
            metrics=metrics,
            lstm_params=lstm,
            n_bootstraps=n_bootstraps,
            n_runs=n_runs,
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


if __name__ == "__main__":
    cfg = build_config()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"profile={cfg['profile']}  device={device}")
    print(f"config: {cfg}")

    # STEP 1: Load the eICU base dataset (dev=True caps to ~1000 patients;
    # dev=False loads the full DB -- the "full" profile).
    base_dataset = eICUDataset(root=cfg["eicu_root"], tables=["diagnosis"], dev=cfg["dev"])

    # STEP 2: Apply the eICU EHR generation task. One fitted SampleDataset gives
    # a single shared code vocabulary, so all per-hospital client models are
    # weight-compatible. Each sample carries a passthrough ``hospital_id``.
    sample_dataset = base_dataset.set_task(EHRGenerationEICU(min_visits=MIN_VISITS))
    vocab_size = sample_dataset.input_processors["visits"].vocab_size()
    print(f"Total samples: {len(sample_dataset)}  code vocab: {vocab_size}")

    sample = sample_dataset[0]
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Hospital ID: {sample.get('hospital_id')}")
    print(f"  Visits tensor shape: {tuple(sample['visits'].shape)}")

    # STEP 3: Partition by hospital -> FedAvg clients + a pooled real test set.
    # If a frozen cohort manifest is given, use exactly its hospitals (in order)
    # so every baseline trains on the identical client set; otherwise fall back
    # to the top-K largest hospitals.
    cohort = None
    if cfg.get("cohort_file"):
        with open(cfg["cohort_file"]) as f:
            manifest = json.load(f)
        cohort = [h["hospital_id"] for h in manifest["hospitals"]]
        print(f"Using frozen cohort from {cfg['cohort_file']}: {cohort}")
    clients, client_tests, pooled_train, pooled_test, info = partition_by_hospital(
        sample_dataset,
        field="hospital_id",
        max_clients=cfg["max_hospitals"],
        min_samples=cfg["min_hospital_samples"],
        test_frac=cfg["test_frac"],
        seed=SEED,
        cohort=cohort,
    )
    print(f"\nHospitals (FedAvg clients): {info}")
    print(f"pooled_train={len(pooled_train)}  pooled_test={len(pooled_test)}")

    # STEP 4: Initialize HALO (small config for the dev subset) and train it with
    # FedAvg across the hospital clients. epochs= is the local epochs per round.
    save_dir = f"_outputs/halo_fed_{cfg['profile']}_save"
    model = HALO(
        dataset=sample_dataset,
        embed_dim=cfg["embed_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_ctx=cfg["n_ctx"],
        batch_size=cfg["batch_size"],
        epochs=cfg["local_epochs"],
        lr=cfg["lr"],
        save_dir=save_dir,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params} parameters")

    # Optional TensorBoard writer for per-hospital train-loss curves (cheap;
    # scalars only). Lazy import so tensorboard stays an optional dependency.
    writer = None
    if not cfg["no_tb"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_logdir = cfg["tb_logdir"] or os.path.join(save_dir, "tb")
            writer = SummaryWriter(log_dir=tb_logdir)
            print(f"TensorBoard logging to {tb_logdir}")
        except ImportError:
            print("tensorboard not installed; skipping TB logging "
                  "(pip install tensorboard, or pass --no-tb)")

    # Checkpoint the aggregated weights every --ckpt-every rounds; --resume
    # continues from the last completed round (a timed-out job loses few rounds).
    ckpt_path = os.path.join(save_dir, "fedavg_state.pt")
    try:
        run_fedavg(model, clients, n_rounds=cfg["n_rounds"], device=device,
                   ckpt_path=ckpt_path, ckpt_every=cfg["ckpt_every"],
                   resume=cfg["resume"], writer=writer,
                   log_every_epochs=cfg["log_every_epochs"])
    finally:
        if writer is not None:
            writer.close()

    # STEP 5: Generate synthetic patients from the aggregated model.
    synthetic = model.generate(num_samples=cfg["num_synth"], device=device)
    print(f"\nGenerated {len(synthetic)} synthetic patients (first 3):")
    for patient in synthetic[:3]:
        print(f"  {patient['patient_id']}: {len(patient['visits'])} visits")

    # STEP 6: GLOBAL evaluation -- score the aggregated model once against the
    # pooled real cohort (the union of every hospital's train/test slices). We
    # default to privacy metrics because this task is unconditional (no labels);
    # to enable the utility metrics, set METRICS and pass a matching label_fn to
    # the real and synthetic frames (see pyhealth/tasks/generate_ehr.py).
    index_to_code = {
        v: k for k, v in sample_dataset.input_processors["visits"].code_vocab.items()
    }

    eval_cfg = {
        "sample_cap": cfg["eval_sample_cap"],
        "lstm": cfg["eval_lstm"],
        "n_bootstraps": cfg["eval_n_bootstraps"],
        "n_runs": cfg["eval_n_runs"],
    }

    print("\n=== Global metrics (aggregated model vs pooled cohort) ===")
    global_results = evaluate_run(
        pooled_train, pooled_test, synthetic, index_to_code,
        metrics=cfg["metrics"], label="global", eval_cfg=eval_cfg,
    )
    if global_results:
        print("\nGlobal generative metrics (mean +/- std):")
        print_metrics(global_results)

    # STEP 7: PER-CLIENT evaluation -- score the SAME global model's synthetic
    # data against each hospital's own real train/test slices. This exposes how
    # evenly the one federated model serves heterogeneous (non-IID) hospitals:
    # a hospital whose distribution is under-represented in the average will show
    # worse fidelity / different privacy numbers here than the pooled view.
    print("\n=== Per-client metrics (same global model, each hospital's data) ===")
    per_client: Dict[str, Dict[str, tuple]] = {}
    for hid in clients:
        res = evaluate_run(
            clients[hid], client_tests[hid], synthetic, index_to_code,
            metrics=cfg["metrics"], label=f"hosp {hid}", eval_cfg=eval_cfg,
        )
        if res:
            per_client[hid] = res
    print("\nPer-hospital generative metrics (mean +/- std):")
    print_client_table(per_client)
 