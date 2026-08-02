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

Set ``--regime {fedavg,fedavg_ft,centralized,local}`` to switch between FedAvg and
three comparison regimes: ``centralized`` pools every hospital's data into one
model (the "no privacy wall" upper bound), ``local`` trains one independent model
per hospital with no aggregation (the "no collaboration" lower bound), and
``fedavg_ft`` fine-tunes the FedAvg global model on each hospital (personalization,
between the two). All regimes share the same partition, vocabulary, per-owner
compute budget, and evaluation, so their metrics are directly comparable.

The federated helpers (``partition_by_hospital``, ``average_state_dicts``,
``run_fedavg``) and the baseline helpers (``train_centralized``, ``train_local``,
``finetune_local``) are inlined below so this example is self-contained. Defaults
are sized for a quick dev/smoke run on one GPU.
"""

import argparse
import json
import os
import statistics
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
    p.add_argument("--profile", choices=sorted(PROFILES), default=None,
                   help="run scale preset (default: smoke, or the --config YAML's "
                        "'profile' if it sets one)")
    p.add_argument("--config",
                   help="YAML file of config overrides, applied ON TOP of "
                        "--profile and BELOW explicit CLI flags. Lets a sweep set "
                        "any knob -- including ones without a dedicated flag (lr, "
                        "embed_dim, n_heads, n_layers, n_ctx, batch_size, "
                        "weighting, eval_* ...). See examples/fedpyhealth/sweeps/.")
    p.add_argument("--weighting", choices=["sample", "uniform"], default=None,
                   help="FedAvg aggregation weighting: 'sample' (default) weights "
                        "each hospital by its sample count; 'uniform' weights every "
                        "hospital equally. Only affects fedavg / fedavg_ft.")
    p.add_argument("--regime",
                   choices=["fedavg", "fedavg_ft", "centralized", "local"],
                   default=None,
                   help="training regime: fedavg (default) trains one model with "
                        "FedAvg across hospital clients; fedavg_ft additionally "
                        "fine-tunes the global model on each hospital "
                        "(personalization); centralized pools every hospital into "
                        "one model; local trains one independent model per "
                        "hospital (no aggregation). centralized and local are "
                        "non-federated baselines that bracket fedavg.")
    p.add_argument("--ft-epochs", type=int,
                   help="fedavg_ft only: local fine-tuning epochs per hospital "
                        "after FedAvg (default: same as --local-epochs)")
    p.add_argument("--eicu-root", default=None, help="path to eICU CRD root")
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
    p.add_argument("--resume", action="store_true", default=None,
                   help="resume FedAvg from the on-disk checkpoint if present")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="checkpoint frequency in rounds (1=every round); the "
                        "final round is always checkpointed (default: 1)")
    p.add_argument("--cohort-file",
                   help="JSON manifest from select_cohort.py: use exactly its "
                        "hospitals as clients instead of the top-K by size")
    p.add_argument("--tb-logdir",
                   help="TensorBoard log dir (default: <save_dir>/tb). "
                        "Logs per-hospital train-loss curves.")
    p.add_argument("--no-tb", action="store_true", default=None,
                   help="disable TensorBoard logging entirely")
    p.add_argument("--log-every-epochs", type=int, default=None,
                   help="log per-hospital loss every N local epochs (last epoch "
                        "of each round is always logged; default: 1)")
    p.add_argument("--run-name",
                   help="readable run id for artifacts + compare_runs labelling "
                        "(default: auto from regime/E/R/cohort/metrics)")
    return p


def make_run_name(cfg: dict) -> str:
    """Readable, filesystem-safe run id derived from the run's key knobs.

    e.g. ``fedavg_E2_R39_standard_8_utility`` or ``centralized_E2_R39_...``.
    Encodes the regime, local_epochs (E), n_rounds (R), the cohort manifest stem
    (or ``topk`` when none), and the metric group -- enough to tell co-located
    runs apart and to drive ``compare_runs.py`` labelling. Two runs that differ
    in any of these get distinct artifact dirs (save/checkpoint/TensorBoard), so
    they never clobber -- including a fedavg run and its centralized/local
    baselines on the same cohort.
    """
    if cfg.get("cohort_file"):
        cohort = os.path.splitext(os.path.basename(cfg["cohort_file"]))[0]
    else:
        cohort = f"top{cfg['max_hospitals']}"
    name = (f"{cfg['regime']}_E{cfg['local_epochs']}_R{cfg['n_rounds']}"
            f"_{cohort}_{cfg['metrics']}")
    # fedavg_ft varies by ft_epochs at a fixed budget -- encode it so an ft sweep's
    # runs get distinct artifacts/results instead of clobbering one another.
    if cfg.get("regime") == "fedavg_ft":
        name += f"_ft{cfg.get('ft_epochs', cfg['local_epochs'])}"
    # Tag uniform-weighted FedAvg runs so they get distinct artifacts/results and
    # never clobber the sample-weighted ones (sample weighting keeps the old name).
    if cfg.get("weighting", "sample") == "uniform":
        name += "_uniform"
    return name


def _load_yaml(path: str) -> dict:
    """Load a config-override YAML into a dict (empty file -> {})."""
    import yaml  # lazy: only needed when --config is used
    with open(path) as f:
        ydict = yaml.safe_load(f) or {}
    if not isinstance(ydict, dict):
        raise ValueError(
            f"Config YAML {path} must be a mapping of knob: value, got "
            f"{type(ydict).__name__}.")
    return ydict


def _overlay(cfg: dict, ydict: dict) -> dict:
    """Overlay a mapping onto ``cfg`` (overlay wins). ``eval_lstm`` deep-merges so
    a sweep can tweak one LSTM knob without restating the whole block."""
    out = dict(cfg)
    for k, v in ydict.items():
        if k == "eval_lstm" and isinstance(v, dict) \
                and isinstance(out.get("eval_lstm"), dict):
            out["eval_lstm"] = {**out["eval_lstm"], **v}
        else:
            out[k] = v
    return out


def build_config(argv: List[str] = None) -> dict:
    """Resolve a config dict by precedence: profile defaults < --config YAML <
    explicit CLI flags. Any knob absent from all three falls back to its built-in
    default below. This layering is what lets a sweep ship one YAML per run while
    a CLI flag can still override a single knob for a quick one-off."""
    args = _build_arg_parser().parse_args(argv)

    # Layer 1: profile defaults. The effective profile is the CLI --profile if
    # given, else the YAML's 'profile', else smoke -- resolved BEFORE building the
    # base dict so a YAML 'profile: full' actually pulls the full-profile defaults
    # (embed_dim, batch_size, eval_* ...), not just the keys the YAML restates.
    ycfg = _load_yaml(args.config) if args.config else {}
    profile = args.profile or ycfg.get("profile") or "smoke"
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    cfg = dict(PROFILES[profile])
    cfg["profile"] = profile

    # Layer 2: YAML overrides profile defaults.
    cfg = _overlay(cfg, ycfg)

    # Layer 3: explicit CLI flags override YAML. Every override arg defaults to
    # None, so "not None" reliably means "the user passed it on the command line".
    cli = {
        "regime": args.regime, "weighting": args.weighting,
        "eicu_root": args.eicu_root, "resume": args.resume,
        "ckpt_every": args.ckpt_every, "cohort_file": args.cohort_file,
        "tb_logdir": args.tb_logdir, "no_tb": args.no_tb,
        "log_every_epochs": args.log_every_epochs,
        "max_hospitals": args.max_hospitals,
        "min_hospital_samples": args.min_hospital_samples,
        "n_rounds": args.n_rounds, "local_epochs": args.local_epochs,
        "num_synth": args.num_synth, "metrics": args.metrics, "dev": args.dev,
        "ft_epochs": args.ft_epochs, "run_name": args.run_name,
    }
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v

    # Fall-back defaults for knobs not set by profile / YAML / CLI.
    cfg.setdefault("regime", "fedavg")
    cfg.setdefault("weighting", "sample")
    cfg.setdefault("eicu_root", EICU_ROOT)
    cfg.setdefault("resume", False)
    cfg.setdefault("ckpt_every", 1)
    cfg.setdefault("cohort_file", None)
    cfg.setdefault("tb_logdir", None)
    cfg.setdefault("no_tb", False)
    cfg.setdefault("log_every_epochs", 1)
    # ft_epochs defaults to track local_epochs (only meaningful for fedavg_ft).
    cfg.setdefault("ft_epochs", cfg["local_epochs"])

    if cfg["weighting"] not in ("sample", "uniform"):
        raise ValueError(
            f"weighting must be 'sample' or 'uniform', got {cfg['weighting']!r}")

    # run_name LAST so it reflects the fully resolved regime/E/R/metrics/cohort.
    if not cfg.get("run_name"):
        cfg["run_name"] = make_run_name(cfg)
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
    weighting: str = "sample",
    writer=None,
    log_every_epochs: int = 1,
    log: Callable[[str], None] = print,
) -> object:
    """Train ``model`` with FedAvg across ``clients`` for ``n_rounds`` rounds.

    The model holds the *global* weights between rounds; ``model._epochs`` (set
    at construction) is the number of local epochs per round. Returns the model
    holding the final aggregated weights.

    ``weighting`` controls how client updates are combined each round:
    ``"sample"`` (classic FedAvg) weights each hospital by its train-set size, so
    larger hospitals pull the global model more; ``"uniform"`` gives every
    hospital equal weight regardless of size, which can help when a few large
    hospitals would otherwise dominate a heterogeneous federation.

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

    log(f"FedAvg: {len(client_ids)} clients, sizes={sizes}, rounds={n_rounds}, "
        f"weighting={weighting}")
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
            weights.append(1.0 if weighting == "uniform" else float(sizes[cid]))
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
# Non-federated baselines: centralized (pool everything) and local-only (one
# independent model per hospital, no aggregation). They bracket FedAvg -- the
# centralized model is the "no privacy wall" upper bound, the local-only models
# the "no collaboration" lower bound. For a fair comparison every regime gets the
# same per-owner compute budget: FedAvg runs n_rounds * local_epochs passes over
# each client's data, so the baseline models are built with epochs set to that
# product before being handed to these helpers.
# ----------------------------------------------------------------------------
def train_centralized(model, pooled_train, device: str = "cpu", writer=None,
                      log: Callable[[str], None] = print) -> object:
    """Train ONE model on the pooled (all-hospital) train data. Returns it.

    With ``writer`` set, the mean train loss is logged under ``loss_train/pooled``
    each epoch (matching the FedAvg per-hospital curves' cumulative x-axis).
    """
    log(f"Centralized: 1 model on pooled train (n={len(pooled_train)})")

    def _on_epoch_end(epoch, mean_loss):
        if writer is not None:
            writer.add_scalar("loss_train/pooled", mean_loss, epoch)
        log(f"  centralized  epoch {epoch + 1}  loss={mean_loss:.4f}")

    model.train_model(pooled_train, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end)
    return model


def train_local(
    build_model: Callable[[], object],
    clients: Dict[str, object],
    device: str = "cpu",
    writer=None,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """Train an INDEPENDENT model per hospital on its own data (no averaging).

    ``build_model`` is a zero-arg factory returning a fresh, untrained model, so
    each hospital gets its own weights. Returns ``{hospital_id: trained_model}``.
    Per-hospital train loss is logged under ``loss_train/hospital_<id>`` so the
    curves line up with the FedAvg run's in TensorBoard.
    """
    models: Dict[str, object] = {}
    for hid, train_subset in clients.items():
        log(f"Local-only: training hospital {hid} (n={len(train_subset)})")
        m = build_model()

        def _on_epoch_end(epoch, mean_loss, hid=hid):
            if writer is not None:
                writer.add_scalar(f"loss_train/hospital_{hid}", mean_loss, epoch)

        m.train_model(train_subset, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end)
        models[hid] = m
    return models


def finetune_local(
    global_state: Dict[str, torch.Tensor],
    build_model: Callable[[], object],
    clients: Dict[str, object],
    device: str = "cpu",
    writer=None,
    log: Callable[[str], None] = print,
) -> Dict[str, object]:
    """FedAvg + fine-tuning: personalize the shared global model per hospital.

    Each hospital warm-starts from the final FedAvg weights (``global_state``)
    and trains a few more local epochs on its own data. This sits between FedAvg
    (one shared model for everyone) and local-only (no shared knowledge at all):
    it keeps the federated model's cross-hospital signal but lets each hospital
    specialize. ``build_model`` must return a fresh model whose epochs = the
    desired fine-tuning epochs. Returns ``{hospital_id: fine_tuned_model}``.
    """
    models: Dict[str, object] = {}
    for hid, train_subset in clients.items():
        log(f"FedAvg+FT: fine-tuning hospital {hid} from global "
            f"(n={len(train_subset)})")
        m = build_model()
        m.load_state_dict(global_state)  # warm start from the federated global

        def _on_epoch_end(epoch, mean_loss, hid=hid):
            if writer is not None:
                writer.add_scalar(f"loss_ft/hospital_{hid}", mean_loss, epoch)

        m.train_model(train_subset, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end)
        models[hid] = m
    return models


def generate_local(
    models: Dict[str, object],
    sizes: Dict[str, int],
    num_synth: int,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Generate synthetic patients from each hospital's own local model.

    Each hospital's share of ``num_synth`` is proportional to its train size, so
    the pooled synthetic set mirrors the real hospital mix (matching how FedAvg
    sample-count-weights its clients). Returns
    ``(pooled_synthetic, {hospital_id: synthetic})`` -- the pooled set is scored
    globally, each hospital's own set against its own data.
    """
    total = float(sum(sizes.values())) or 1.0
    pooled: List[Dict] = []
    per_hosp: Dict[str, List[Dict]] = {}
    for hid, m in models.items():
        share = max(1, int(round(num_synth * sizes[hid] / total)))
        syn = m.generate(num_samples=share, device=device)
        per_hosp[hid] = syn
        pooled.extend(syn)
        log(f"  [local] hospital {hid}: generated {len(syn)} synthetic")
    return pooled, per_hosp


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


def _metrics_to_json(results: Dict[str, tuple]) -> Dict[str, dict]:
    """Turn a {name: (mean, std)} metric dict into JSON-friendly nested dicts."""
    return {name: {"mean": float(mean), "std": float(std)}
            for name, (mean, std) in results.items()}


def _macro_average(per_client: Dict[str, Dict[str, tuple]]) -> Dict[str, dict]:
    """Macro-average each metric across hospitals (unweighted mean of the
    per-hospital means; ``between_hospital_std`` = spread across hospitals).
    Mirrors how compare_runs.py collapses the per-hospital table to one number."""
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
            "between_hospital_std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "n_hospitals": len(vals),
        }
    return out


def save_results_json(path: str, cfg: dict, info: Dict[str, dict],
                      num_params: int, global_results, per_client) -> str:
    """Write one run's raw stats to ``path`` as JSON.

    Captures the full config, the partition (per-hospital train/test sizes), the
    model size, the global (pooled) metrics, every per-hospital metric, and a
    macro-averaged summary -- enough to rebuild any comparison table offline and
    to aggregate a whole sweep without re-parsing SLURM logs. Metric values are
    stored as {"mean", "std"} pairs."""
    # Promoted to the top level so a sweep leaderboard can read them cheaply.
    knob_keys = ("profile", "regime", "weighting", "local_epochs", "n_rounds",
                 "ft_epochs", "lr", "embed_dim", "n_heads", "n_layers", "n_ctx",
                 "batch_size", "num_synth", "metrics", "max_hospitals",
                 "min_hospital_samples", "test_frac", "cohort_file")
    payload = {
        "run_name": cfg["run_name"],
        "regime": cfg["regime"],
        "weighting": cfg.get("weighting", "sample"),
        "key_knobs": {k: cfg.get(k) for k in knob_keys},
        "config": cfg,
        "num_params": int(num_params),
        "partition": info,
        "global_metrics": _metrics_to_json(global_results) if global_results else None,
        "per_hospital_metrics": {h: _metrics_to_json(r)
                                 for h, r in per_client.items() if r},
        "macro_avg_metrics": _macro_average(per_client),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


if __name__ == "__main__":
    cfg = build_config()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"profile={cfg['profile']}  device={device}")
    print(f"RUN_NAME: {cfg['run_name']}")
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

    # STEP 4: Build + train the generator under the chosen regime. Every regime
    # gets the SAME apples-to-apples compute budget -- `total_epochs` =
    # n_rounds * local_epochs passes over each owner's data:
    #   * fedavg      -- ONE model, FedAvg across hospital clients (local_epochs
    #                    per round x n_rounds rounds), weighted averaging.
    #   * fedavg_ft   -- SPLITS the budget: (total_epochs - ft_epochs) passes of
    #                    FedAvg, then ft_epochs of per-hospital fine-tuning. So its
    #                    total passes EQUAL fedavg/local/centralized (not total +
    #                    ft). Global eval uses the shared model; per-client eval
    #                    uses each hospital's own fine-tuned model. Sits between
    #                    fedavg (ft_epochs=0) and local (ft_epochs=total_epochs).
    #   * centralized -- ONE model on the pooled (all-hospital) data for
    #                    total_epochs epochs. Upper bound: no privacy wall.
    #   * local       -- one INDEPENDENT model per hospital on its own data for
    #                    total_epochs epochs, never aggregated. Lower bound: no
    #                    cross-hospital collaboration.
    # save_dir is per-run-name (regime + ft_epochs are encoded in run_name) so
    # regimes keep separate checkpoints + TB logs and never clobber each other.
    regime = cfg["regime"]
    total_epochs = cfg["n_rounds"] * cfg["local_epochs"]
    save_dir = f"_outputs/{cfg['run_name']}_save"

    # Apples-to-apples budget split for fedavg_ft: spend ft_epochs of the budget on
    # per-hospital fine-tuning and the REST on FedAvg, so total passes/hospital ==
    # total_epochs (same as every other regime). Plain fedavg spends all on FedAvg.
    fed_rounds = cfg["n_rounds"]
    if regime == "fedavg_ft":
        fed_budget = total_epochs - cfg["ft_epochs"]
        if fed_budget < cfg["local_epochs"]:
            raise ValueError(
                f"ft_epochs ({cfg['ft_epochs']}) leaves no room for FedAvg: need "
                f"ft_epochs <= total_epochs - local_epochs "
                f"(= {total_epochs - cfg['local_epochs']}). Raise n_rounds or "
                f"lower ft_epochs.")
        fed_rounds = fed_budget // cfg["local_epochs"]
        spent = fed_rounds * cfg["local_epochs"] + cfg["ft_epochs"]
        if spent != total_epochs:
            print(f"[budget] fedavg_ft passes/hospital = {spent} "
                  f"(target {total_epochs}; off by {total_epochs - spent} from "
                  f"integer rounds -- pick ft_epochs so total_epochs - ft_epochs "
                  f"is divisible by local_epochs for an exact match).")

    def build_model(epochs: int):
        """Fresh HALO sharing the global vocab (so weights are compatible)."""
        return HALO(
            dataset=sample_dataset,
            embed_dim=cfg["embed_dim"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            n_ctx=cfg["n_ctx"],
            batch_size=cfg["batch_size"],
            epochs=epochs,
            lr=cfg["lr"],
            save_dir=save_dir,
        )

    # Optional TensorBoard writer for train-loss curves (cheap; scalars only).
    # Lazy import so tensorboard stays an optional dependency.
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

    sizes = {hid: len(clients[hid]) for hid in clients}
    # client_synth[hid] = the synthetic set scored against hospital `hid` in the
    # per-client eval (STEP 7). For fedavg/centralized that's the single global
    # model's output (shared by all); for local it's that hospital's OWN model.
    client_synth: Dict[str, List[Dict]] = {}
    try:
        if regime == "local":
            print(f"\nRegime: local-only -- {len(clients)} independent models, "
                  f"{total_epochs} epochs each")
            models = train_local(lambda: build_model(total_epochs), clients,
                                 device=device, writer=writer)
            num_params = sum(p.numel()
                             for p in next(iter(models.values())).parameters())
            print(f"Each model: {num_params} parameters")
            # STEP 5: each local model generates its own (size-weighted) share.
            synthetic, client_synth = generate_local(
                models, sizes, cfg["num_synth"], device=device)
        else:
            # centralized trains the single model for the full budget; fedavg and
            # fedavg_ft use local_epochs per round (the FedAvg local step).
            epochs = total_epochs if regime == "centralized" else cfg["local_epochs"]
            model = build_model(epochs)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel initialized with {num_params} parameters")
            if regime == "centralized":
                print(f"Regime: centralized -- 1 model on pooled data, "
                      f"{total_epochs} epochs")
                train_centralized(model, pooled_train, device=device, writer=writer)
            else:  # fedavg or fedavg_ft -- both start with a FedAvg run
                print(f"Regime: {regime} -- {len(clients)} clients, "
                      f"{cfg['local_epochs']} local epochs x {fed_rounds} rounds"
                      + (f", then {cfg['ft_epochs']} fine-tuning epochs/hospital "
                         f"(total {total_epochs} passes/hospital)"
                         if regime == "fedavg_ft" else ""))
                # Checkpoint aggregated weights every --ckpt-every rounds; --resume
                # continues from the last completed round (fedavg only).
                ckpt_path = os.path.join(save_dir, "fedavg_state.pt")
                run_fedavg(model, clients, n_rounds=fed_rounds, device=device,
                           ckpt_path=ckpt_path, ckpt_every=cfg["ckpt_every"],
                           resume=cfg["resume"], weighting=cfg["weighting"],
                           writer=writer,
                           log_every_epochs=cfg["log_every_epochs"])
            # STEP 5: generate synthetic patients from the single global/server
            # model. For fedavg_ft this stays the SHARED global model's output, so
            # the global eval (STEP 6) is directly comparable to plain fedavg.
            synthetic = model.generate(num_samples=cfg["num_synth"], device=device)
            client_synth = {hid: synthetic for hid in clients}
            if regime == "fedavg_ft":
                # Personalize: fine-tune the global model on each hospital, then
                # score each hospital against its OWN fine-tuned model in STEP 7.
                global_state = _snapshot(model)
                ft_models = finetune_local(
                    global_state, lambda: build_model(cfg["ft_epochs"]),
                    clients, device=device, writer=writer)
                _, client_synth = generate_local(
                    ft_models, sizes, cfg["num_synth"], device=device)
    finally:
        if writer is not None:
            writer.close()

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

    # STEP 7: PER-CLIENT evaluation -- score each hospital's synthetic data against
    # its own real train/test slices. For fedavg/centralized that synthetic is the
    # single global model's output (so this exposes how evenly one model serves
    # heterogeneous, non-IID hospitals); for local it is that hospital's OWN
    # model's output (so this is each local baseline scored on its home turf).
    print("\n=== Per-client metrics (each hospital's synthetic vs its own data) ===")
    per_client: Dict[str, Dict[str, tuple]] = {}
    for hid in clients:
        res = evaluate_run(
            clients[hid], client_tests[hid], client_synth[hid], index_to_code,
            metrics=cfg["metrics"], label=f"hosp {hid}", eval_cfg=eval_cfg,
        )
        if res:
            per_client[hid] = res
    print("\nPer-hospital generative metrics (mean +/- std):")
    print_client_table(per_client)

    # STEP 8: persist the raw stats (config + partition + per-hospital metrics +
    # macro summary) so a sweep can be aggregated without re-parsing SLURM logs.
    results_path = save_results_json(
        os.path.join("_outputs", "results", f"{cfg['run_name']}.json"),
        cfg, info, num_params, global_results, per_client,
    )
    print(f"\nSaved results -> {results_path}")
 