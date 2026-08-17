"""Example: federated synthetic-EHR generation on eICU with HALO + FedAvg.

eICU is multi-hospital -- every patient carries a ``hospitalid`` -- which makes it
the natural vehicle for a *federated* generative-EHR setup. This single script
runs the whole pipeline end-to-end:

1. Read the cohort cache built by ``utils/cohort.py`` -- 8 hospitals x 3 folds of
   Parquet, plus the pinned code vocabulary. This never touches eICU, so every
   regime provably trains and scores on identical data.
2. Train ONE HALO generator with FedAvg across the hospital clients (train each
   client locally, then sample-count-weighted average the weights each round).
3. Generate synthetic patients from the aggregated model.
4. Evaluate the synthetic data globally against the pooled real cohort with the
   generative metrics suite, and per hospital against that hospital's own data,
   to show how evenly one federated model serves heterogeneous sites.

Set ``--regime {fedavg,fedavg_ft,centralized,local}`` to switch between FedAvg and
three comparison regimes: ``centralized`` pools every hospital's data into one
model (the "no privacy wall" upper bound), ``local`` trains one independent model
per hospital with no aggregation (the "no collaboration" lower bound), and
``fedavg_ft`` fine-tunes the FedAvg global model on each hospital (personalization,
between the two). All regimes share the same partition, vocabulary, per-owner
compute budget, and evaluation, so their metrics are directly comparable.

The federated helpers (``average_state_dicts``,
``run_fedavg``) and the baseline helpers (``train_centralized``, ``train_local``,
``finetune_local``) are inlined below so this example is self-contained. Defaults
are sized for a quick tiny run on one GPU.
"""

import argparse
import json
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from utils.cohort import (
    DEFAULT_CACHE_DIR,
    load_clients,
    load_fold,
    load_manifest,
    manifest_sha256,
)
# The three jobs of an experiment live in three modules, so a change to one
# does not drag the others' cost with it:
#   train.py     trains the generator(s) and checkpoints them   (~18h, GPU)
#   generate.py  samples synthetic patients from a checkpoint    (~13m, GPU)
#   eval.py      scores those patients against real data         (CPU)
# This file still calls the other two at the end of a run so a single job
# produces a complete result, but each is independently runnable against a
# finished run's save_dir.
from generate import proportional_pool, write_synthetic
from eval import evaluate_and_save
from pyhealth.models import HALO

# --- configuration ----------------------------------------------------------
# The cohort, the split and the vocabulary all come from the cache; nothing in
# this file knows where eICU lives. SEED covers weight init only.
SEED = 0

# Profiles select the *scale* of a run, NOT the data. Both read the same cohort
# cache, so a "tiny" run exercises the same code path and the same 8 clients as
# the real thing -- it just does less of it (~5-10 min).
#
# What a tiny run does NOT give you is numbers. At num_synth=64 prevalence
# resolves only to 1/64, so Test 1's R^2 is quantization noise. Tiny answers
# "does this run end to end", never "is this any good".
#
# "full" is a suggested production-scale run -- tune to your compute budget; it
# is a starting point, not a validated config. Pick one with
# `--profile {tiny,full}` (default: tiny), and override any individual knob with
# its `--<knob>` flag (see _build_arg_parser).
PROFILES: Dict[str, dict] = {
    "tiny": dict(
        # 2 rounds, not 1: fedavg_ft SPLITS the budget between FedAvg and
        # per-hospital fine-tuning, so a 1-epoch total budget leaves nothing for
        # the FedAvg half and the regime refuses to run. Two is the smallest
        # budget under which all four regimes are exercisable -- which is the
        # entire point of a smoke profile.
        n_rounds=2,                # FedAvg communication rounds
        local_epochs=1,            # local training epochs per client per round
        num_synth=64,              # too few to measure prevalence -- see note above
        synth_per_hospital=32,     # smoke-sized; the full profile is what matters
        metrics="privacy",         # "privacy" | "utility" | "all" (utility needs label_fn)
        # small HALO config: enough to prove the wiring, not to learn anything
        embed_dim=64, n_heads=2, n_layers=2, n_ctx=20, batch_size=16, lr=1e-4,
        # downsized metric evaluator
        eval_sample_cap=30,
        eval_lstm=dict(embed_dim=16, hidden_dim=16, batch_size=16, epochs=3),
        eval_n_bootstraps=3, eval_n_runs=2,
    ),
    "full": dict(
        n_rounds=50,               # FedAvg needs many rounds to converge
        local_epochs=2,
        # Sized from the frozen cohort, not guessed: a synthetic set resolves
        # prevalence only to 1/num_synth, and strat8's rarest frozen code sits at
        # 0.00067 (2 of 3005 patients at hospital 420). 10 expected occurrences
        # of that code needs 10/0.00067 ~= 15000. 5000 is the deliberate
        # compromise: it clears 10 expected occurrences for 75% of the cohort's
        # (hospital, code) rare pairs (the median rare code gets ~22), leaving
        # the deepest quartile quantization-limited. Report Test 1's rare R^2
        # banded by support, or raise this, before claiming the deep tail.
        num_synth=5000,
        # Every hospital generates this many, regardless of its real size --
        # the point of the comparison is whether a 79-patient site can still
        # draw usable synthetic data from a collaboratively trained generator.
        # At 2000, hospital 438 trains a downstream classifier on 25x the real
        # data it holds, which is the claim being tested.
        #
        # Generation is not what makes this number expensive (~6 min across 8
        # hospitals; ~5000 patients per 2 min including eval passes, against an
        # ~18h train). The cost is downstream: test2 runs uncapped, so every
        # extra synthetic patient is extra classifier training across 16
        # per-hospital arms x 4 mask folds. 2000 holds per-fold training volume
        # to ~53.6k records vs ~85.6k at 4000.
        #
        # Resolution caveat, unchanged in kind from 5000: 1/2000 = 5e-4 clears
        # hospital 438's rarest code (0.0179) by a wide margin but leaves
        # hospital 420's deepest tail (6.7e-4) quantization-limited. Band Test
        # 1's rare metrics by support before claiming the deep tail.
        synth_per_hospital=2000,
        # NOT "utility"/"all". That group is compute_mle, whose downstream task
        # is hard-coded to next-visit prediction -- degenerate here, because this
        # cohort's median patient has ONE unit stay (p50=1, p90=2), so most
        # patients yield no train pair and the score would describe the ~30%
        # multi-stay minority while looking like a cohort-wide number. The real
        # ML-efficacy evidence is Test 2 (test2_rare_efficacy.py), which scores
        # rare-code recovery with a per-hospital classifier instead.
        metrics="privacy",
        # larger HALO config for full vocabulary / longer sequences
        embed_dim=256, n_heads=4, n_layers=4, n_ctx=50, batch_size=64, lr=1e-4,
        # heavier metric evaluator for tighter confidence intervals
        eval_sample_cap=200,
        eval_lstm=dict(embed_dim=64, hidden_dim=64, batch_size=64, epochs=10),
        eval_n_bootstraps=10, eval_n_runs=5,
    ),
}

# Helper-function default args reference the tiny profile so the FedAvg helpers
# stay importable standalone with sane defaults.
_TINY = PROFILES["tiny"]
N_ROUNDS = _TINY["n_rounds"]
METRICS = _TINY["metrics"]


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI: pick a profile, then optionally override individual knobs."""
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--profile", choices=sorted(PROFILES), default=None,
                   help="run scale preset (default: tiny, or the --config YAML's "
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
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR,
                   help="cohort cache directory built by utils/cohort.py. It "
                        "defines the hospitals, the 70/10/20 split and the "
                        "pinned vocabulary; nothing here reads eICU")
    # per-knob overrides (default None -> keep the profile's value)
    p.add_argument("--n-rounds", type=int)
    p.add_argument("--local-epochs", type=int)
    p.add_argument("--num-synth", type=int,
                   help="size of the proportional pooled synthetic set (the "
                        "one scored against the real pooled cohort)")
    p.add_argument("--synth-per-hospital", type=int,
                   help="synthetic patients generated PER HOSPITAL by the "
                        "multi-model regimes (local, fedavg_ft), identical for "
                        "every hospital regardless of its real size. Defaults "
                        "to --num-synth")
    p.add_argument("--metrics", choices=["privacy", "utility", "all"])
    p.add_argument("--resume", action="store_true", default=None,
                   help="resume from the on-disk checkpoint if present (all "
                        "four regimes checkpoint; local keeps one file per "
                        "hospital and skips the ones already finished)")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="checkpoint frequency, in rounds for fedavg/fedavg_ft "
                        "and in epochs for centralized/local (1=every one); the "
                        "last is always checkpointed (default: 1)")
    p.add_argument("--tb-logdir",
                   help="TensorBoard log dir (default: <save_dir>/tb). "
                        "Logs per-hospital train-loss curves.")
    p.add_argument("--no-tb", action="store_true", default=None,
                   help="disable TensorBoard logging entirely")
    p.add_argument("--log-every-epochs", type=int, default=None,
                   help="log per-hospital loss every N local epochs (last epoch "
                        "of each round is always logged; default: 1)")
    p.add_argument("--run-name",
                   help="readable run id for artifacts + results.py labelling "
                        "(default: auto from regime/E/R/cohort/metrics)")
    return p


def make_run_name(cfg: dict) -> str:
    """Readable, filesystem-safe run id derived from the run's key knobs.

    e.g. ``fedavg_E2_R39_strat8_utility`` or ``centralized_E2_R39_...``.
    Encodes the regime, local_epochs (E), n_rounds (R), the cohort manifest stem,
    and the metric group -- enough to tell co-located
    runs apart and to drive ``results.py`` labelling. Two runs that differ
    in any of these get distinct artifact dirs (save/checkpoint/TensorBoard), so
    they never clobber -- including a fedavg run and its centralized/local
    baselines on the same cohort.
    """
    cohort = os.path.basename(str(cfg["cohort_cache"]).rstrip("/"))
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
    # given, else the YAML's 'profile', else tiny -- resolved BEFORE building the
    # base dict so a YAML 'profile: full' actually pulls the full-profile defaults
    # (embed_dim, batch_size, eval_* ...), not just the keys the YAML restates.
    ycfg = _load_yaml(args.config) if args.config else {}
    profile = args.profile or ycfg.get("profile") or "tiny"
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
        "resume": args.resume, "cohort_cache": args.cohort_cache,
        "ckpt_every": args.ckpt_every,
        "tb_logdir": args.tb_logdir, "no_tb": args.no_tb,
        "log_every_epochs": args.log_every_epochs,
        "n_rounds": args.n_rounds, "local_epochs": args.local_epochs,
        "num_synth": args.num_synth, "metrics": args.metrics,
        "synth_per_hospital": args.synth_per_hospital,
        "ft_epochs": args.ft_epochs, "run_name": args.run_name,
    }
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v

    # Fall-back defaults for knobs not set by profile / YAML / CLI.
    cfg.setdefault("regime", "fedavg")
    cfg.setdefault("weighting", "sample")
    cfg.setdefault("resume", False)
    cfg.setdefault("cohort_cache", DEFAULT_CACHE_DIR)
    cfg.setdefault("ckpt_every", 1)
    cfg.setdefault("tb_logdir", None)
    cfg.setdefault("no_tb", False)
    cfg.setdefault("log_every_epochs", 1)
    # ft_epochs defaults to track local_epochs (only meaningful for fedavg_ft).
    cfg.setdefault("ft_epochs", cfg["local_epochs"])
    # Per-hospital synthetic count. Falls back to num_synth so an old YAML that
    # never heard of this knob still produces a self-consistent run.
    cfg.setdefault("synth_per_hospital", cfg["num_synth"])

    if cfg["weighting"] not in ("sample", "uniform"):
        raise ValueError(
            f"weighting must be 'sample' or 'uniform', got {cfg['weighting']!r}")
    if not cfg.get("cohort_cache"):
        raise ValueError(
            "--cohort-cache is required: every regime must train on the same "
            "cohort and the same split, or the comparison between them means "
            "nothing. Build one with `python utils/cohort.py`.")

    # run_name LAST so it reflects the fully resolved regime/E/R/metrics/cohort.
    if not cfg.get("run_name"):
        cfg["run_name"] = make_run_name(cfg)
    return cfg


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


def _fingerprint(sizes: Dict[str, int], split_sha: str = None) -> dict:
    """Identity of a partition, so resume refuses a mismatched checkpoint.

    Client sizes alone are not enough: two different frozen splits of the same
    cohort have identical sizes, so resuming across them would silently train
    on one partition and evaluate on another. ``split_sha`` pins the exact
    manifest.
    """
    return {
        "sizes": {cid: int(n) for cid, n in sorted(sizes.items())},
        "split_sha": split_sha,
    }


def _save_ckpt(path: str, completed: int, global_state, fingerprint: dict,
               key: str = "completed_rounds"):
    """Atomically write the FedAvg checkpoint (tmp file + rename).

    The tmp-then-rename keeps the checkpoint valid even if the job is killed
    mid-write -- ``os.replace`` is atomic on POSIX, so we never leave a
    half-written file that a later resume would choke on.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {key: completed, "global_state": global_state,
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
    split_sha: str = None,
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
    fingerprint = _fingerprint(sizes, split_sha)

    global_state = _snapshot(model)
    start_round = 0
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Checkpoint {ckpt_path} was written for a different partition "
                f"(client sizes and/or frozen split differ); refusing to "
                f"resume. Delete it, or point --cohort-file at the manifest it "
                f"was trained against."
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
def _load_epoch_ckpt(path: str, fingerprint: dict, log) -> Tuple[dict, int]:
    """Read an epoch-granular checkpoint, refusing one from a different setup.

    Returns ``(state, completed_epochs)``, or ``(None, 0)`` when there is
    nothing to resume from.
    """
    if not os.path.exists(path):
        return None, 0
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("fingerprint") != fingerprint:
        raise ValueError(
            f"Checkpoint {path} was written for a different partition or "
            f"cohort; refusing to resume. Delete it, or point --cohort-cache at "
            f"the cache it was trained against."
        )
    log(f"  resumed {path} at epoch {ckpt['completed_epochs']}")
    return ckpt["global_state"], int(ckpt["completed_epochs"])


def train_centralized(model, pooled_train, device: str = "cpu", writer=None,
                      log: Callable[[str], None] = print,
                      total_epochs: int = None, ckpt_path: str = None,
                      ckpt_every: int = 1, resume: bool = False,
                      fingerprint: dict = None) -> object:
    """Train ONE model on the pooled (all-hospital) train data. Returns it.

    Checkpointing is epoch-granular rather than round-granular (there are no
    rounds here): the weights are written every ``ckpt_every`` epochs, and
    ``resume=True`` continues from the last completed one. Without it a job
    killed at the wall clock loses the entire run, which is exactly what
    happened at ~12h before this existed.

    Only weights are saved -- ``train_model`` builds a fresh optimizer per call,
    so a resumed run restarts Adam's moment estimates and is not bit-identical
    to an uninterrupted one. The same caveat applies to the FedAvg checkpoints.
    """
    total_epochs = total_epochs or int(getattr(model, "_epochs", 1))
    start_epoch = 0
    if resume and ckpt_path:
        state, start_epoch = _load_epoch_ckpt(ckpt_path, fingerprint, log)
        if state is not None:
            model.load_state_dict(state)
    if start_epoch >= total_epochs:
        log(f"Centralized: checkpoint already has {start_epoch} >= "
            f"{total_epochs} epochs; skipping training")
        return model

    log(f"Centralized: 1 model on pooled train (n={len(pooled_train)}), "
        f"epochs {start_epoch}->{total_epochs}")

    def _on_epoch_end(epoch, mean_loss):
        done = start_epoch + epoch + 1
        if writer is not None:
            writer.add_scalar("loss_train/pooled", mean_loss, done - 1)
        log(f"  centralized  epoch {done}/{total_epochs}  loss={mean_loss:.4f}")
        if ckpt_path and (done % ckpt_every == 0 or done == total_epochs):
            _save_ckpt(ckpt_path, done, _snapshot(model), fingerprint,
                       key="completed_epochs")

    # Resuming means training only what is left, so the model is rebuilt with
    # the remaining epoch count rather than the original budget.
    model._epochs = total_epochs - start_epoch
    model.train_model(pooled_train, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end)
    return model


def train_local(
    build_model: Callable[[], object],
    clients: Dict[str, object],
    device: str = "cpu",
    writer=None,
    log: Callable[[str], None] = print,
    total_epochs: int = None,
    ckpt_dir: str = None,
    ckpt_every: int = 1,
    resume: bool = False,
    fingerprint: dict = None,
) -> Dict[str, object]:
    """Train an INDEPENDENT model per hospital on its own data (no averaging).

    ``build_model`` is a zero-arg factory returning a fresh, untrained model, so
    each hospital gets its own weights. Returns ``{hospital_id: trained_model}``.

    Checkpoints go to ONE FILE PER HOSPITAL (``local_<hid>.pt``) rather than one
    growing blob holding all eight: a finished hospital is never rewritten, and
    a corrupted file costs one hospital instead of the whole run. Resume skips
    hospitals whose file is complete and continues the one that was in flight --
    which matters most here, since this regime is 8 sequential trainings and has
    the longest exposure to a wall-clock kill.
    """
    models: Dict[str, object] = {}
    for hid, train_subset in clients.items():
        epochs = total_epochs or int(getattr(build_model(), "_epochs", 1))
        path = os.path.join(ckpt_dir, f"local_{hid}.pt") if ckpt_dir else None
        m = build_model()

        start_epoch = 0
        if resume and path:
            state, start_epoch = _load_epoch_ckpt(path, fingerprint, log)
            if state is not None:
                m.load_state_dict(state)
        if start_epoch >= epochs:
            log(f"Local-only: hospital {hid} already complete "
                f"({start_epoch}/{epochs} epochs); skipping")
            models[hid] = m
            continue

        log(f"Local-only: training hospital {hid} (n={len(train_subset)}), "
            f"epochs {start_epoch}->{epochs}")

        def _on_epoch_end(epoch, mean_loss, hid=hid, path=path,
                          start_epoch=start_epoch, epochs=epochs, m=m):
            done = start_epoch + epoch + 1
            if writer is not None:
                writer.add_scalar(f"loss_train/hospital_{hid}", mean_loss,
                                  done - 1)
            if path and (done % ckpt_every == 0 or done == epochs):
                _save_ckpt(path, done, _snapshot(m), fingerprint,
                           key="completed_epochs")

        m._epochs = epochs - start_epoch
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
    ckpt_dir: str = None,
    fingerprint: dict = None,
) -> Dict[str, object]:
    """FedAvg + fine-tuning: personalize the shared global model per hospital.

    Each hospital warm-starts from the final FedAvg weights (``global_state``)
    and trains a few more local epochs on its own data. This sits between FedAvg
    (one shared model for everyone) and local-only (no shared knowledge at all):
    it keeps the federated model's cross-hospital signal but lets each hospital
    specialize. ``build_model`` must return a fresh model whose epochs = the
    desired fine-tuning epochs. Returns ``{hospital_id: fine_tuned_model}``.

    Each fine-tuned model is written to ``<ckpt_dir>/ft_<hospital>.pt``. Without
    that, ``fedavg_ft`` is the one regime whose per-hospital generators exist
    only in RAM, so re-generating its synthetic data at a different size means
    re-running fine-tuning instead of loading a checkpoint -- see generate.py.

    Args:
        global_state: Final FedAvg weights, the warm start for every hospital.
        build_model: Returns a fresh model with epochs = fine-tuning epochs.
        clients: ``{hospital_id: train_subset}``.
        device: Torch device.
        writer: Optional TensorBoard writer.
        log: Progress sink.
        ckpt_dir: Where to persist each fine-tuned model. None skips saving.
        fingerprint: Partition identity stamped into each checkpoint so one
            written against a different cohort or split is refused on load.

    Returns:
        ``{hospital_id: fine_tuned_model}``.
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
        if ckpt_dir:
            path = os.path.join(ckpt_dir, f"ft_{hid}.pt")
            _save_ckpt(path, 1, _snapshot(m), fingerprint,
                       key="completed_finetune")
            log(f"  fine-tuned hospital {hid} -> {path}")
    return models


def generate_local(
    models: Dict[str, object],
    sizes: Dict[str, int],
    num_synth: int,
    per_hospital_n: int,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Generate synthetic patients from each hospital's own local model.

    Every hospital generates the SAME ``per_hospital_n`` patients regardless of
    how much real data it holds. Sizing a hospital's synthetic output by its own
    train size would hard-code the assumption that a small site deserves less
    synthetic data -- which is precisely the hypothesis federated generation is
    supposed to test. A tiny hospital that benefits from collaboration should be
    able to draw large amounts of usable synthetic data despite holding little
    real data, and it can: generation is a sampling loop, not training.

    Two pooled views come out of the same generated patients:

    - **proportional** (returned as the pooled set): hospital ``h`` contributes
      ``num_synth * sizes[h] / sum(sizes)`` patients, so the mix mirrors the real
      cohort. This is the only view comparable against the real pooled cohort --
      pooled prevalence is a hospital-weighted average, so scoring a differently
      mixed set against real pooled data penalises even a perfect generator.
    - **uniform**: every hospital contributes equally. Recover it downstream by
      concatenating the per-hospital sets (see ``load_synthetic``); it is a
      representation view, not a fidelity metric, and must not be scored against
      the real pooled cohort.

    Args:
        models: ``{hospital_id: generator}``.
        sizes: ``{hospital_id: n_train}``, used only for the proportional mix.
        num_synth: Size of the proportional pooled set.
        per_hospital_n: Patients generated per hospital, identical for all.
        device: Torch device for generation.
        log: Progress sink.

    Returns:
        ``(pooled_proportional, {hospital_id: synthetic})``.

    Raises:
        ValueError: If ``per_hospital_n`` is too small for the largest
            hospital's proportional share, which would silently under-fill the
            pooled set and skew its mix.
    """
    total = float(sum(sizes.values())) or 1.0
    per_hosp: Dict[str, List[Dict]] = {}
    for hid, m in models.items():
        syn = m.generate(num_samples=per_hospital_n, device=device)
        per_hosp[hid] = syn
        log(f"  [local] hospital {hid}: generated {len(syn)} synthetic")

    # The proportional set is a prefix subsample of what we already generated --
    # generation is i.i.d., so a prefix is a valid sample and no second pass is
    # needed. The mixing rule itself lives in generate.py so the in-run path and
    # the regenerate-from-checkpoint path cannot drift apart.
    pooled = proportional_pool(per_hosp, sizes, num_synth)
    log(f"  [local] pooled_proportional: {len(pooled)} patients "
        f"(uniform view = {sum(len(v) for v in per_hosp.values())})")
    return pooled, per_hosp


if __name__ == "__main__":
    cfg = build_config()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"profile={cfg['profile']}  device={device}")
    print(f"RUN_NAME: {cfg['run_name']}")
    print(f"config: {cfg}")

    # STEPS 1-3: read the cohort straight off the cache -- the hospitals, the
    # 70/10/20 split and the pinned vocabulary all come from there, so nothing
    # below touches eICU and every regime provably sees identical data. The
    # cache's own build step verified that its Parquet files reproduce the eICU
    # tensors byte-for-byte, which is what makes this shortcut safe.
    #
    # Train on train, score on VAL while the pipeline is still under
    # development. The test fold is never read here: tuning against val
    # therefore cannot leak into the final numbers.
    EVAL_FOLD = "val"
    cache = cfg["cohort_cache"]
    manifest = load_manifest(cache)
    split_sha = manifest_sha256(manifest)
    cohort = list(manifest["hospitals"])
    print(f"Cohort '{manifest.get('cohort_name')}' from {cache}: {cohort}")
    print(f"  split {manifest['fracs']}, guaranteed "
          f"{manifest['guaranteed_folds']}, rare <= "
          f"{manifest['rare_prevalence_max']} per hospital")

    clients_by_fold = load_clients(cache, folds=("train", EVAL_FOLD))
    clients = {hid: f["train"] for hid, f in clients_by_fold.items()}
    client_tests = {hid: f[EVAL_FOLD] for hid, f in clients_by_fold.items()}
    pooled_train = load_fold(cache, "train")
    pooled_test = load_fold(cache, EVAL_FOLD)
    sample_dataset = pooled_train          # any fold: they share one processor
    vocab_size = sample_dataset.input_processors["visits"].vocab_size()

    info = {hid: {"n_total": manifest["per_hospital"][hid]["n_total"],
                  "n_train": len(clients[hid]),
                  f"n_{EVAL_FOLD}": len(client_tests[hid])}
            for hid in cohort}
    for hid in cohort:
        want = {f: manifest["per_hospital"][hid][f"n_{f}"]
                for f in ("train", EVAL_FOLD)}
        got = {"train": len(clients[hid]), EVAL_FOLD: len(client_tests[hid])}
        if got != want:
            raise ValueError(
                f"hospital {hid}: loaded {got} but the cache manifest says "
                f"{want}; the Parquet files and the manifest disagree.")

    sample = sample_dataset[0]
    print(f"\nCode vocab: {vocab_size}   sample visits tensor: "
          f"{tuple(sample['visits'].shape)}")

    # A synthetic set of size S can only express prevalences in multiples of
    # 1/S. If the rarest scored code sits below that grid, PrevVal_Rare_* is
    # measuring quantization, not fidelity -- and it looks like a real number.
    smallest = min(h["min_rare_prevalence"]
                   for h in manifest["per_hospital"].values())
    floor = int(np.ceil(10 / smallest)) if smallest > 0 else 0
    if cfg["num_synth"] < floor:
        print(
            f"\n!! WARNING: num_synth={cfg['num_synth']} resolves prevalence "
            f"only to {1.0 / max(1, cfg['num_synth']):.6f}, but the rarest "
            f"scored code has prevalence {smallest:.6f}. PrevVal_Rare_* will "
            f"be dominated by quantization noise.\n!! Use --num-synth >= "
            f"{floor} for this cohort.\n"
        )
    print(f"\nHospitals (FedAvg clients): {info}")
    print(f"pooled_train={len(pooled_train)}  "
          f"pooled_{EVAL_FOLD}={len(pooled_test)} (scoring fold)")

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

    # Stamp the config NOW, not at the end. generate.py and eval.py both need to
    # know the architecture and knobs a checkpoint was written for, and a run
    # that times out mid-training still has usable checkpoints -- so the config
    # has to survive a job that never reaches its final save.
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

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
    # One partition identity shared by every regime's checkpoints, so a
    # checkpoint written against a different cohort or split is refused rather
    # than silently resumed onto the wrong data.
    fingerprint = _fingerprint(sizes, split_sha)
    # client_synth[hid] = the synthetic set scored against hospital `hid` in the
    # per-client eval (STEP 7). For fedavg/centralized that's the single global
    # model's output (shared by all); for local it's that hospital's OWN model.
    client_synth: Dict[str, List[Dict]] = {}
    try:
        if regime == "local":
            print(f"\nRegime: local-only -- {len(clients)} independent models, "
                  f"{total_epochs} epochs each")
            models = train_local(lambda: build_model(total_epochs), clients,
                                 device=device, writer=writer,
                                 total_epochs=total_epochs, ckpt_dir=save_dir,
                                 ckpt_every=cfg["ckpt_every"],
                                 resume=cfg["resume"], fingerprint=fingerprint)
            num_params = sum(p.numel()
                             for p in next(iter(models.values())).parameters())
            print(f"Each model: {num_params} parameters")
            # STEP 5: every local model generates the SAME number of patients;
            # the pooled set is then a proportional subsample of those.
            synthetic, client_synth = generate_local(
                models, sizes, cfg["num_synth"], cfg["synth_per_hospital"],
                device=device)
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
                train_centralized(
                    model, pooled_train, device=device, writer=writer,
                    total_epochs=total_epochs,
                    ckpt_path=os.path.join(save_dir, "centralized_state.pt"),
                    ckpt_every=cfg["ckpt_every"], resume=cfg["resume"],
                    fingerprint=fingerprint)
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
                           log_every_epochs=cfg["log_every_epochs"],
                           split_sha=split_sha)
            # STEP 5: generate synthetic patients from the single global/server
            # model. For fedavg_ft this stays the SHARED global model's output, so
            # the global eval (STEP 6) is directly comparable to plain fedavg.
            # A single generator serves every hospital, so its output has no
            # natural per-site ceiling -- it can emit as much as asked. Size it
            # as the FEDERATION TOTAL (8 x the per-hospital count) so
            # "centralized's synthetic output" and "the federation's combined
            # output" are the same magnitude, rather than comparing one site's
            # share against a whole pooled system.
            want = cfg["num_synth"]
            if regime in ("centralized", "fedavg"):
                want = max(want, len(clients) * cfg["synth_per_hospital"])
            full = model.generate(num_samples=want, device=device)
            client_synth = {hid: full for hid in clients}
            # The POOLED set stays num_synth for every regime, so Test 1's
            # prevalence resolution (1/num_synth) is identical across arms and
            # its numbers stay comparable. Only the per-hospital sets scale.
            synthetic = full[:cfg["num_synth"]]
            if regime == "fedavg_ft":
                # Personalize: fine-tune the global model on each hospital, then
                # score each hospital against its OWN fine-tuned model in STEP 7.
                global_state = _snapshot(model)
                ft_models = finetune_local(
                    global_state, lambda: build_model(cfg["ft_epochs"]),
                    clients, device=device, writer=writer,
                    ckpt_dir=save_dir, fingerprint=fingerprint)
                _, client_synth = generate_local(
                    ft_models, sizes, cfg["num_synth"],
                    cfg["synth_per_hospital"], device=device)
    finally:
        if writer is not None:
            writer.close()

    print(f"\nGenerated {len(synthetic)} synthetic patients (first 3):")
    for patient in synthetic[:3]:
        print(f"  {patient['patient_id']}: {len(patient['visits'])} visits")

    # STEP 6: persist the synthetic data. Written BEFORE scoring so a job that
    # dies (or is killed) during evaluation still leaves something re-scorable
    # by eval.py -- the expensive half is already paid for by this point.
    # fedavg_ft's client_synth comes from its 8 fine-tuned models, so only
    # centralized and fedavg genuinely share one generator.
    synth_path = write_synthetic(
        save_dir, synthetic, client_synth,
        shared_generator=regime in ("centralized", "fedavg"))
    print(f"Saved synthetic data -> {synth_path}")

    # STEP 7: score it. The whole suite lives in eval.py and is runnable
    # standalone against this save_dir, so a metric fix costs a CPU job rather
    # than another ~18h of GPU. Deleting this call is all it takes to make
    # train.py training-only.
    index_to_code = {
        v: k for k, v in sample_dataset.input_processors["visits"].code_vocab.items()
    }
    results_path = evaluate_and_save(
        cfg, manifest, clients, client_tests, pooled_train, pooled_test,
        synthetic, client_synth, index_to_code, info,
        num_params=num_params, split_sha=split_sha,
    )
    print(f"\nSaved results -> {results_path}")
 