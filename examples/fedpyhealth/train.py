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
        # batch_size 256. Measured on hilo8_random, doubling the batch is a
        # real speedup rather than a bigger number: this pipeline is bound by
        # per-batch Python overhead, not by the model.
        #
        #   batch   GPU util   GPU mem      of an A100's 40960 MB
        #     64      23%       1260 MB       3%
        #    128      62%       2194 MB       5%
        #    256       -       ~4300 MB     ~11%   (extrapolated)
        #
        # Memory is nowhere near the limit -- 1024 would still leave half the
        # card free. The real cost is gradient steps at the small sites, and it
        # is worth stating plainly because it is invisible in any throughput
        # number. Batches per epoch on hilo8_random:
        #
        #             458(1832)  449(1001)  277(624)  358(194)  429(142)
        #    b=128       15          8          5         2         2
        #    b=256        8          4          3         1         1
        #
        # So at 256 the two smallest hospitals take ONE optimizer step per
        # epoch. They also run the full budget rather than early-stopping
        # (es_fallback=fixed, their val folds are under es_min_val_patients),
        # so the full budget is now a handful of steps. Read any small-site
        # result with that in mind.
        #
        # The fix, if that matters: cap the batch PER CLIENT, e.g.
        # min(batch_size, ceil(n_train / 4)), so 458 runs at 256 while 429 runs
        # at 36 and every site keeps >=4 steps. That needs run_fedavg to build a
        # model per client instead of sharing one -- see notes/TODO.md A3.
        #
        # Raising the batch spends gradient steps to buy throughput, so if val
        # loss plateaus higher than it did at 128, raise lr before blaming the
        # data (the ft sweep showed 3e-4 is stable for this model).
        embed_dim=256, n_heads=4, n_layers=4, n_ctx=50, batch_size=256, lr=1e-4,
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
    p.add_argument("--rare-upweight", action="store_true", default=None,
                   help="weight each patient's loss by the rarity of the codes "
                        "they carry AT THEIR OWN HOSPITAL: w = 1 + sum of "
                        "1/local_prevalence over that site's rare codes the "
                        "patient has, capped at the site's 99th percentile and "
                        "normalised to mean 1. Applies to every regime, keyed "
                        "on patient_id, so the pooled centralized arm still "
                        "weights each patient by their own site's view. Off by "
                        "default; run_name gets a _rw suffix so weighted and "
                        "unweighted runs never share a save_dir.")
    p.add_argument("--irm-rho", type=float, default=None,
                   help="IRMv1 penalty weight. 0 (default) trains the plain "
                        "risk. Above 0 each client optimises risk + rho * "
                        "penalty, where the penalty is the squared gradient of "
                        "its own loss w.r.t. a dummy classifier fixed at 1.0 -- "
                        "zero when w=1 is already optimal for that ENVIRONMENT, "
                        "and environments here are hospitals. The penalty is "
                        "per-environment and summed, so it is fully separable "
                        "and FedAvg needs NO extra communication: only each "
                        "client's local loss changes. Needs a second-order "
                        "backward, so expect ~2x step cost. Typical post-warmup "
                        "values are 1e2-1e4; it needs a sweep. run_name gets an "
                        "_irm suffix.")
    p.add_argument("--irm-warmup", type=int, default=None,
                   help="rounds (fedavg) or epochs (centralized/local) to hold "
                        "rho at 1.0 before jumping to --irm-rho. IRM cannot be "
                        "trained by switching a large penalty on at step 0: the "
                        "model takes the trivial invariant solution and never "
                        "fits the data, which then looks like 'IRM does "
                        "nothing'. Default 0 applies rho immediately -- set it "
                        "to roughly a third of the budget for a real run.")
    p.add_argument("--irm-unweighted-envs", action="store_true", default=None,
                   help="documentation flag, on by design and not yet "
                        "switchable: the IRM penalty is summed across "
                        "hospitals UNWEIGHTED, so a 142-patient site has the "
                        "same voice as an 1,832-patient one. That is "
                        "deliberate -- IRM is about environments, not samples -- "
                        "and it deviates from --weighting sample, which governs "
                        "only the FedAvg aggregation.")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="checkpoint frequency, in rounds for fedavg/fedavg_ft "
                        "and in epochs for centralized/local (1=every one); the "
                        "last is always checkpointed (default: 1)")
    p.add_argument("--snapshot-every", type=int, default=None,
                   help="centralized only: ALSO keep a numbered, never-"
                        "overwritten copy of the weights every N epochs "
                        "(centralized_epoch<NNNN>.pt). --ckpt-every rewrites one "
                        "rolling file, which is right for resuming and useless "
                        "for asking what the model looked like earlier -- and "
                        "that question is the only way to test whether long "
                        "training degrades SAMPLING while val loss keeps "
                        "falling. Costs ~25 MB per snapshot.")
    p.add_argument("--tb-logdir",
                   help="TensorBoard log dir (default: <save_dir>/tb). "
                        "Logs per-hospital train-loss curves.")
    p.add_argument("--no-tb", action="store_true", default=None,
                   help="disable TensorBoard logging entirely")
    p.add_argument("--log-every-epochs", type=int, default=None,
                   help="log per-hospital loss every N local epochs (last epoch "
                        "of each round is always logged; default: 1)")
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                   default=None,
                   help="run the full fixed budget instead of stopping when "
                        "validation loss stops improving")
    p.add_argument("--es-patience", type=int,
                   help="rounds (fedavg) or epochs (others) of no val "
                        "improvement before stopping (default: 5)")
    p.add_argument("--es-min-steps", type=int,
                   help="never stop before this many rounds/epochs (default: 3)")
    p.add_argument("--es-min-val-patients", type=int,
                   help="a hospital needs this many val patients before its "
                        "own val loss is trusted to stop it; smaller sites "
                        "fall back to --es-fallback (default: 30)")
    p.add_argument("--es-fallback", choices=["fixed", "train_plateau"],
                   help="what a too-small site does instead: run the fixed "
                        "budget (default) or stop on a train-loss plateau")
    p.add_argument("--eval-fold", choices=["val", "test"],
                   help="fold the in-run scoring reports on (default: test, "
                        "since val is now the model-selection fold)")
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
    # Rare-upweighting changes what the generator optimises, so it must not
    # share a save_dir, a checkpoint or a results row with the plain run.
    if cfg.get("rare_upweight"):
        name += "_rw"
    # Same reasoning for IRM, plus rho in the name: a rho sweep is the point of
    # the experiment, so two rho values must not land on one save_dir.
    if cfg.get("irm_rho", 0.0) > 0:
        name += f"_irm{cfg['irm_rho']:g}"
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
        "snapshot_every": args.snapshot_every,
        "rare_upweight": args.rare_upweight,
        "tb_logdir": args.tb_logdir, "no_tb": args.no_tb,
        "log_every_epochs": args.log_every_epochs,
        "n_rounds": args.n_rounds, "local_epochs": args.local_epochs,
        "num_synth": args.num_synth, "metrics": args.metrics,
        "synth_per_hospital": args.synth_per_hospital,
        "ft_epochs": args.ft_epochs, "run_name": args.run_name,
        "irm_rho": args.irm_rho, "irm_warmup": args.irm_warmup,
        "early_stop": args.early_stop, "es_patience": args.es_patience,
        "es_min_steps": args.es_min_steps,
        "es_min_val_patients": args.es_min_val_patients,
        "es_fallback": args.es_fallback, "eval_fold": args.eval_fold,
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
    cfg.setdefault("snapshot_every", 0)      # 0 = keep no numbered snapshots
    cfg.setdefault("rare_upweight", False)
    cfg.setdefault("irm_rho", 0.0)
    cfg.setdefault("irm_warmup", 0)
    cfg.setdefault("tb_logdir", None)
    cfg.setdefault("no_tb", False)
    cfg.setdefault("log_every_epochs", 1)
    # ft_epochs defaults to track local_epochs (only meaningful for fedavg_ft).
    cfg.setdefault("ft_epochs", cfg["local_epochs"])
    # Per-hospital synthetic count. Falls back to num_synth so an old YAML that
    # never heard of this knob still produces a self-consistent run.
    cfg.setdefault("synth_per_hospital", cfg["num_synth"])

    # Early stopping. On by default: with it off, every regime runs a fixed
    # budget and the small sites get the same handful of gradient steps that
    # made fedavg_ft look worse than local.
    cfg.setdefault("early_stop", True)
    cfg.setdefault("es_patience", 5)
    cfg.setdefault("es_min_delta", 0.0)
    # Never stop inside the first few steps: a generator's val loss sits nearly
    # flat while it learns the code marginals, and patience alone would call
    # that convergence.
    cfg.setdefault("es_min_steps", 3)
    # Below this many val patients the fold's mean moves more from which
    # patients landed in it than from what the model learned. On hilo8_random
    # this puts 6 of 8 sites on val and leaves 358 (28) and 429 (20) on the
    # fallback.
    cfg.setdefault("es_min_val_patients", 30)
    cfg.setdefault("es_fallback", "fixed")       # or "train_plateau"
    # Fold the in-run scoring reports on. val is the model-selection fold now
    # that early stopping reads it, so reporting moved to test to keep the two
    # apart; test1/test2 take --fold test to match.
    cfg.setdefault("eval_fold", "test")

    if cfg["es_fallback"] not in ("fixed", "train_plateau"):
        raise ValueError(
            f"es_fallback must be 'fixed' or 'train_plateau', got "
            f"{cfg['es_fallback']!r}")
    if cfg["eval_fold"] not in ("val", "test"):
        raise ValueError(
            f"eval_fold must be 'val' or 'test', got {cfg['eval_fold']!r}")
    if cfg["early_stop"] and cfg["eval_fold"] == "val":
        raise ValueError(
            "early_stop selects the model on val, so scoring on val too would "
            "report a number the model was tuned against. Use eval_fold=test, "
            "or set early_stop=false.")

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


# --------------------------------------------------------------------------- #
# Validation loss and early stopping                                           #
# --------------------------------------------------------------------------- #
# Val loss is computed HERE rather than by handing ``val_dataset`` to
# ``HALO.train_model``. That method writes a best-model checkpoint to
# ``save_dir/halo_model`` whenever val improves AND loads that same file at the
# top of every call -- and all eight FedAvg clients share one ``save_dir``, so
# passing a val set would make client B silently warm-start from client A's
# weights instead of the broadcast global. FedAvg would stop being FedAvg with
# nothing raised. See notes/TODO.md A1.
#
# The stop itself rides on ``on_epoch_end``: returning False from the callback
# ends that model's training after the current epoch, so the criterion lives
# here and HALO's loop stays unaware of it.
def val_loss(model, dataset, device: str, batch_size: int = None) -> float:
    """Mean per-batch validation loss, the same quantity HALO trains on.

    Mirrors the forward pass in ``HALO.train_model`` exactly -- same encoding,
    same ``pos_loss_weight`` -- so the number is comparable to the training
    loss printed alongside it rather than a differently-normalised cousin.

    Args:
        model: A HALO wrapper (must already be on ``device``).
        dataset: The fold to score, as a ``SampleDataset``.
        device: Torch device string.
        batch_size: Loader batch size; defaults to the model's own.

    Returns:
        Mean loss over batches, or ``nan`` for an empty dataset (a hospital
        with no val patients is a real possibility on a small site, and a NaN
        that propagates into "did it improve?" is safer than a fabricated 0.0).
    """
    from pyhealth.datasets import get_dataloader

    if len(dataset) == 0:
        return float("nan")
    loader = get_dataloader(dataset,
                            batch_size=batch_size or int(model._batch_size),
                            shuffle=False)
    model.halo_model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            visits = batch["visits"].to(device)
            ehr, mask = model._encode_visits(visits)
            loss, _, _ = model.halo_model(
                ehr, position_ids=None, ehr_labels=ehr, ehr_masks=mask,
                pos_loss_weight=model.config.pos_loss_weight)
            losses.append(float(loss.item()))
    model.halo_model.train()
    return float(np.mean(losses)) if losses else float("nan")


class EarlyStopper:
    """Track a minimised score, remember the best weights, say when to stop.

    Deliberately keeps the best state in RAM rather than on disk: the on-disk
    path is the one HALO already uses, and writing to it is what would break
    FedAvg (see above). A HALO state dict at the ``full`` profile is ~36 MB, so
    one snapshot per model is cheap next to what training already holds.

    Args:
        patience: Consecutive non-improving steps tolerated before stopping.
        min_delta: How much better a score must be to count as an improvement.
            Guards against declaring victory on numerical noise.
        min_steps: Never stop before this many steps, however flat the curve.
            A generator's val loss can sit flat for the first epochs while the
            model learns the code marginals, and stopping there would report a
            near-untrained model as "converged".

    Attributes:
        best_score: Lowest score seen.
        best_state: Weights snapshot from the step that produced it.
        best_step: 1-indexed step of the best score.
        stopped_early: True if ``step`` ever returned True.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 min_steps: int = 0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_steps = int(min_steps)
        self.best_score = float("inf")
        self.best_state = None
        self.best_step = 0
        self.stopped_early = False
        self.n_steps = 0
        self.n_bad = 0

    def step(self, score: float, state=None) -> bool:
        """Record one step's score. Returns True when training should stop.

        Args:
            score: The quantity being minimised (val loss, usually).
            state: Weights to remember if this step is the best so far. May be
                a dict, or a zero-arg callable returning one -- the callable
                form is only invoked on an improvement, so the ~36 MB snapshot
                is not paid on epochs that turn out not to matter.

        A NaN score (empty val fold) never improves and never counts against
        patience, so a misconfigured fold stalls the criterion instead of
        silently ending the run at step ``patience``.
        """
        self.n_steps += 1
        if score != score:                       # NaN
            return False
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_step = self.n_steps
            self.n_bad = 0
            if state is not None:
                self.best_state = state() if callable(state) else state
        else:
            self.n_bad += 1
        if self.n_steps >= self.min_steps and self.n_bad >= self.patience:
            self.stopped_early = True
            return True
        return False

    def summary(self, unit: str = "epoch") -> str:
        if self.best_state is None and self.best_step == 0:
            return "no validation signal (all scores NaN)"
        how = "early-stopped" if self.stopped_early else "ran to budget"
        return (f"{how} after {self.n_steps} {unit}s; best val "
                f"{self.best_score:.4f} at {unit} {self.best_step}")


# --------------------------------------------------------------------------- #
# GPU accounting                                                               #
# --------------------------------------------------------------------------- #
# Measured on the finished runs, this pipeline barely touches the card it books:
# the 12h FedAvg run averaged 22% GPU utilisation and 1276 MB of an A100's
# 40960 MB, and asked SLURM for 128 GB of host RAM while using 2.9 GB. That is
# not a rounding error -- it means wall-clock is set by everything around the
# compute rather than by the compute, and batch_size could rise a long way
# before memory bites. Sampling during training turns that from post-hoc sacct
# archaeology into something the run itself reports.
_GPU_SAMPLES: List[Tuple[float, float]] = []   # (utilisation %, peak memory MB)


def gpu_snapshot() -> Dict[str, float]:
    """Current GPU utilisation and memory, or {} when there is no GPU.

    Utilisation comes from ``nvidia-smi`` rather than
    ``torch.cuda.utilization()`` because the latter needs pynvml, which is not
    installed in this environment. The call costs a few milliseconds, so it
    belongs at round/epoch boundaries -- never inside the batch loop.
    """
    if not torch.cuda.is_available():
        return {}
    out: Dict[str, float] = {
        "mem_allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "mem_reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "mem_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }
    try:
        free, total = torch.cuda.mem_get_info()
        out["mem_total_mb"] = total / 1e6
        out["mem_used_frac"] = (total - free) / total
    except Exception:                              # noqa: BLE001
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            out["util_pct"] = float(r.stdout.strip().splitlines()[0])
    except Exception:                              # noqa: BLE001
        pass                                        # no nvidia-smi: skip silently
    return out


def log_gpu(writer, step: int, tag: str = "gpu") -> None:
    """Record one GPU sample to TensorBoard and to the run-level summary."""
    snap = gpu_snapshot()
    if not snap:
        return
    _GPU_SAMPLES.append((snap.get("util_pct", float("nan")),
                         snap.get("mem_peak_mb", 0.0)))
    if writer is not None:
        for k, v in snap.items():
            writer.add_scalar(f"{tag}/{k}", v, step)


def print_gpu_summary(log: Callable[[str], None] = print) -> None:
    """Say plainly whether the booked GPU was worth booking."""
    if not _GPU_SAMPLES:
        return
    utils = [u for u, _ in _GPU_SAMPLES if u == u]
    peak = max((m for _, m in _GPU_SAMPLES), default=0.0)
    total = gpu_snapshot().get("mem_total_mb", 0.0)
    log("\n=== GPU usage ===")
    if utils:
        mean_u = sum(utils) / len(utils)
        log(f"  utilisation: mean {mean_u:.0f}%  min {min(utils):.0f}%  "
            f"max {max(utils):.0f}%  ({len(utils)} samples)")
        if mean_u < 40:
            log(f"  -> the GPU idled ~{100 - mean_u:.0f}% of the time. Wall-clock "
                "is set by data loading\n     and per-sample Python work, not by "
                "the model. Raising batch_size via\n     --config is the first "
                "thing to try.")
    if peak and total:
        log(f"  memory: peak {peak:.0f} MB of {total:.0f} MB "
            f"({100 * peak / total:.1f}%)")
        if peak < 0.25 * total:
            log(f"  -> {total / max(peak, 1):.0f}x headroom before memory bites.")


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


def irm_rho_at(step: int, rho: float, warmup: int) -> float:
    """IRMv1's penalty weight for one round/epoch, with the paper's warmup.

    IRM cannot be trained by simply switching a large penalty on: at a big
    ``rho`` the penalty dominates from step zero, the model finds the trivial
    invariant solution (predict the same thing everywhere) and never fits the
    data at all. The published recipe trains at ``rho ~ 1`` for a warmup period
    -- long enough to reach a useful representation -- then jumps to the large
    value. A run that "shows IRM does nothing" without warmup usually never
    fitted in the first place.

    The jump is deliberately a STEP, not a ramp, matching the reference
    implementation. Ramping conflates "the penalty grew" with "training
    progressed" and makes the loss curve unreadable at exactly the point you
    need to see it.

    Args:
        step: Completed rounds (FedAvg) or epochs (centralized/local).
        rho: The post-warmup penalty weight.
        warmup: Steps to hold at 1.0 first. ``0`` applies ``rho`` immediately.

    Returns:
        The penalty weight to pass to ``train_model(irm_rho=...)``.
    """
    if rho <= 0.0:
        return 0.0
    return 1.0 if step < warmup else rho


def log_irm(writer, model, step: int, tag: str) -> None:
    """Write this epoch's risk and penalty as two separate curves.

    Never their sum. ``rho`` is the one IRM hyperparameter that genuinely needs
    a sweep and it cannot be set without seeing the two terms' relative
    magnitude; and IRM's characteristic failure -- the penalty driven to zero by
    a model that has stopped fitting anything -- is invisible in the total,
    which just looks like it is going down.

    ``irm_penalty`` can be NEGATIVE. The estimator is a product of two
    independent half-batch gradients, so a sign disagreement between halves
    reads as negative: that means "no reliable gradient here", not a bug. Expect
    it to be common at the small sites.

    No-ops when the run is not using IRM, so every regime can call it blind.
    """
    if writer is None:
        return
    irm = getattr(model, "last_irm", None)
    if not irm or irm.get("rho", 0.0) <= 0.0:
        return
    writer.add_scalar(f"irm_risk/{tag}", irm["risk"], step)
    writer.add_scalar(f"irm_penalty/{tag}", irm["penalty"], step)
    writer.add_scalar("irm_rho", irm["rho"], step)


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
    val_dataset=None,
    stopper: "EarlyStopper" = None,
    sample_weight_fn=None,
    irm_rho: float = 0.0,
    irm_warmup: int = 0,
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

    Early stopping: with ``val_dataset`` and ``stopper`` given, the aggregated
    global model is scored on that fold after every round and training stops
    when it has not improved for ``stopper.patience`` rounds, returning the BEST
    global state rather than the last. The unit is deliberately the round, not
    the local epoch: cutting a client's local epochs short would give clients
    unequal numbers of gradient steps, which biases the weighted average toward
    whoever trained longer and stops the result being FedAvg. Rounds are the
    only place a global model exists to validate.

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
                f"resume. Delete it, or point --cohort-cache at the cache it "
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
                    # Per hospital: which SITES are non-invariant is itself a
                    # result on a cohort with a 13x size spread.
                    log_irm(writer, model, step, f"hospital_{cid}")

            model.train_model(clients[cid], val_dataset=None, device=device,
                              on_epoch_end=_on_epoch_end,
                              sample_weight_fn=sample_weight_fn,
                              irm_rho=irm_rho_at(r, irm_rho, irm_warmup))
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
        log_gpu(writer, r + 1)

        if val_dataset is not None and stopper is not None:
            vl = val_loss(model, val_dataset, device)
            if writer is not None:
                writer.add_scalar("loss_val/global", vl, r + 1)
            stop = stopper.step(vl, state=global_state)
            log(f"  round {r + 1}: val {vl:.4f}  (best {stopper.best_score:.4f} "
                f"at round {stopper.best_step})")
            if stop:
                log(f"FedAvg: early stop at round {r + 1}/{n_rounds} -- "
                    f"{stopper.summary('round')}")
                break

    # Restoring the best global state is the half of early stopping that is
    # easy to leave out: without it the run stops early AND keeps the worse
    # weights it stopped on, which is strictly worse than not stopping.
    #
    # Two things the obvious version gets wrong, both because `best_step` counts
    # rounds within THIS invocation -- a resumed run builds a fresh stopper, so
    # its counter restarts at zero and is NOT a round number in the run's own
    # timeline:
    #
    #   * the checkpoint is stamped with the FULL budget, not with the best
    #     step. Early stopping means the remaining rounds are deliberately not
    #     wanted, so a resume must skip them rather than train them again. On a
    #     resumed run, stamping `best_step` also moves the checkpoint BACKWARDS
    #     -- a run resuming at round 32 and finishing 18 more would record 18 in
    #     place of 50, and the next resume would redo 32 rounds it already had.
    #     This matches train_centralized, which stamps total_epochs for the same
    #     reason.
    #   * the log reports the ABSOLUTE round, so "best at round 37" means round
    #     37 of the experiment rather than the 5th round of whatever the last
    #     job happened to run.
    if stopper is not None and stopper.best_state is not None:
        best_round = start_round + stopper.best_step
        model.load_state_dict(stopper.best_state)
        if ckpt_path:
            _save_ckpt(ckpt_path, n_rounds, stopper.best_state, fingerprint)
        log(f"FedAvg: restored best global (round {best_round}/{n_rounds}, val "
            f"{stopper.best_score:.4f})")

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
                      fingerprint: dict = None, val_dataset=None,
                      stopper: "EarlyStopper" = None,
                      snapshot_every: int = 0,
                      sample_weight_fn=None,
                      irm_rho: float = 0.0,
                      irm_warmup: int = 0) -> object:
    """Train ONE model on the pooled (all-hospital) train data. Returns it.

    Checkpointing is epoch-granular rather than round-granular (there are no
    rounds here): the weights are written every ``ckpt_every`` epochs, and
    ``resume=True`` continues from the last completed one. Without it a job
    killed at the wall clock loses the entire run, which is exactly what
    happened at ~12h before this existed.

    Only weights are saved -- ``train_model`` builds a fresh optimizer per call,
    so a resumed run restarts Adam's moment estimates and is not bit-identical
    to an uninterrupted one. The same caveat applies to the FedAvg checkpoints.

    With ``val_dataset`` and ``stopper``, the pooled val fold is scored after
    every epoch and training stops when it stops improving; the best weights are
    restored at the end. This regime has the whole cohort's val fold behind it,
    so the signal is the least noisy of the four.
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
            log_gpu(writer, done - 1)
        if ckpt_path and (done % ckpt_every == 0 or done == total_epochs):
            _save_ckpt(ckpt_path, done, _snapshot(model), fingerprint,
                       key="completed_epochs")
        # A numbered copy that nothing later overwrites. The rolling checkpoint
        # answers "where do I resume"; this answers "what did the model look
        # like at epoch N", which is the only way to compare SAMPLING behaviour
        # across training when the loss curve is flat.
        if ckpt_path and snapshot_every and done % snapshot_every == 0:
            snap = os.path.join(os.path.dirname(ckpt_path),
                                f"centralized_epoch{done:04d}.pt")
            _save_ckpt(snap, done, _snapshot(model), fingerprint,
                       key="completed_epochs")
            log(f"  snapshot -> {snap}")

        if val_dataset is None or stopper is None:
            log(f"  centralized  epoch {done}/{total_epochs}  "
                f"loss={mean_loss:.4f}")
            return None
        vl = val_loss(model, val_dataset, device)
        if writer is not None:
            writer.add_scalar("loss_val/pooled", vl, done - 1)
            log_irm(writer, model, done - 1, "pooled")
        stop = stopper.step(vl, state=lambda: _snapshot(model))
        log(f"  centralized  epoch {done}/{total_epochs}  loss={mean_loss:.4f}"
            f"  val={vl:.4f}  (best {stopper.best_score:.4f} @ "
            f"{stopper.best_step})")
        # False is the stop request HALO's loop honours; None keeps going.
        return False if stop else None

    # Resuming means training only what is left, so the model is rebuilt with
    # the remaining epoch count rather than the original budget.
    model._epochs = total_epochs - start_epoch
    model.train_model(pooled_train, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end,
                      sample_weight_fn=sample_weight_fn,
                      # start_epoch offset so a RESUMED run does not restart the
                      # warmup it already finished and re-enter the rho=1 regime.
                      irm_rho=lambda e: irm_rho_at(start_epoch + e, irm_rho,
                                                   irm_warmup))
    if stopper is not None:
        log(f"Centralized: {stopper.summary()}")
        if stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)
            if ckpt_path:
                # Stamped with the FULL budget, not best_step: early stopping
                # means the remaining epochs are deliberately not wanted, and a
                # resume that saw best_step would train them anyway.
                _save_ckpt(ckpt_path, total_epochs, stopper.best_state,
                           fingerprint, key="completed_epochs")
    return model


def make_site_stopper(hid: str, n_val: int, es_cfg: dict,
                      log: Callable[[str], None] = print):
    """Decide how one hospital's training ends. Returns ``(stopper, use_val)``.

    Three outcomes, in order of preference:

    * val fold big enough -> stop on val loss, the criterion that actually
      measures generalisation;
    * too small, ``fallback="train_plateau"`` -> stop when TRAIN loss flattens.
      Weaker by construction: train loss on a 142-record site keeps falling as
      the model memorises it, so this usually runs to budget and is a guard
      against wasted epochs rather than a real convergence test;
    * too small, ``fallback="fixed"`` -> no stopper, run the budget.

    ``(None, False)`` means "train the fixed number of epochs".
    """
    if not es_cfg:
        return None, False
    if n_val >= int(es_cfg["min_val_patients"]):
        log(f"  {hid}: early stopping on val (n_val={n_val})")
        return EarlyStopper(es_cfg["patience"], es_cfg["min_delta"],
                            es_cfg["min_steps"]), True
    if es_cfg.get("fallback") == "train_plateau":
        log(f"  {hid}: n_val={n_val} < {es_cfg['min_val_patients']} -- "
            f"stopping on TRAIN-loss plateau instead")
        return EarlyStopper(es_cfg["patience"], es_cfg["min_delta"],
                            es_cfg["min_steps"]), False
    log(f"  {hid}: n_val={n_val} < {es_cfg['min_val_patients']} -- too noisy "
        f"to stop on; fixed budget")
    return None, False


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
    val_clients: Dict[str, object] = None,
    es_cfg: dict = None,
    sample_weight_fn=None,
    irm_rho: float = 0.0,
    irm_warmup: int = 0,
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

    With ``val_clients`` and ``es_cfg``, each hospital stops on its OWN val fold
    once that fold is large enough to mean anything -- see ``make_site_stopper``
    for what happens when it is not. Per-site and not pooled on purpose: this is
    the no-collaboration baseline, and letting a site borrow another site's
    stopping epoch would quietly make it collaborative.
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

        val_ds = (val_clients or {}).get(hid)
        stopper, use_val = make_site_stopper(
            hid, len(val_ds) if val_ds is not None else 0, es_cfg, log)

        def _on_epoch_end(epoch, mean_loss, hid=hid, path=path,
                          start_epoch=start_epoch, epochs=epochs, m=m,
                          val_ds=val_ds, stopper=stopper, use_val=use_val):
            done = start_epoch + epoch + 1
            if writer is not None:
                writer.add_scalar(f"loss_train/hospital_{hid}", mean_loss,
                                  done - 1)
                log_irm(writer, m, done - 1, f"hospital_{hid}")
                log_gpu(writer, done - 1, tag="gpu_local")
            if path and (done % ckpt_every == 0 or done == epochs):
                _save_ckpt(path, done, _snapshot(m), fingerprint,
                           key="completed_epochs")
            if stopper is None:
                return None
            score = val_loss(m, val_ds, device) if use_val else mean_loss
            if writer is not None and use_val:
                writer.add_scalar(f"loss_val/hospital_{hid}", score, done - 1)
            stop = stopper.step(score, state=lambda: _snapshot(m))
            return False if stop else None

        m._epochs = epochs - start_epoch
        m.train_model(train_subset, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end,
                      sample_weight_fn=sample_weight_fn,
                      irm_rho=lambda e, s0=start_epoch: irm_rho_at(
                          s0 + e, irm_rho, irm_warmup))
        if stopper is not None:
            log(f"  {hid}: {stopper.summary()}")
            if stopper.best_state is not None:
                m.load_state_dict(stopper.best_state)
                if path:
                    # Full budget, not best_step -- see train_centralized.
                    _save_ckpt(path, epochs, stopper.best_state, fingerprint,
                               key="completed_epochs")
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
    val_clients: Dict[str, object] = None,
    es_cfg: dict = None,
    sample_weight_fn=None,
    irm_rho: float = 0.0,
    irm_warmup: int = 0,
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
        val_clients: ``{hospital_id: val_subset}``. With ``es_cfg``, each site
            fine-tunes until its own val loss stops improving instead of for a
            fixed ``ft_epochs``. Safe here in a way it is not inside the
            federated rounds: this runs AFTER the last aggregation, so sites
            taking different numbers of steps never get averaged together.
        es_cfg: Early-stopping settings; see ``make_site_stopper``.

    Returns:
        ``{hospital_id: fine_tuned_model}``.
    """
    models: Dict[str, object] = {}
    for hid, train_subset in clients.items():
        log(f"FedAvg+FT: fine-tuning hospital {hid} from global "
            f"(n={len(train_subset)})")
        m = build_model()
        m.load_state_dict(global_state)  # warm start from the federated global

        val_ds = (val_clients or {}).get(hid)
        stopper, use_val = make_site_stopper(
            hid, len(val_ds) if val_ds is not None else 0, es_cfg, log)

        def _on_epoch_end(epoch, mean_loss, hid=hid, m=m, val_ds=val_ds,
                          stopper=stopper, use_val=use_val):
            if writer is not None:
                writer.add_scalar(f"loss_ft/hospital_{hid}", mean_loss, epoch)
                log_irm(writer, m, epoch, f"ft_{hid}")
                log_gpu(writer, epoch, tag="gpu_ft")
            if stopper is None:
                return None
            score = val_loss(m, val_ds, device) if use_val else mean_loss
            if writer is not None and use_val:
                writer.add_scalar(f"loss_val_ft/hospital_{hid}", score, epoch)
            stop = stopper.step(score, state=lambda: _snapshot(m))
            return False if stop else None

        m.train_model(train_subset, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end,
                      sample_weight_fn=sample_weight_fn,
                      # Fine-tuning runs AFTER the last aggregation, so warmup
                      # is already long over -- rho applies from step 0 here.
                      irm_rho=irm_rho)
        if stopper is not None:
            log(f"  {hid}: {stopper.summary()}")
            # The warm start means epoch 0 can already be the best the site
            # gets: fine-tuning a converged global on 142 records often makes
            # val loss worse immediately, and keeping the best is what stops
            # that from silently becoming the reported fedavg_ft result.
            if stopper.best_state is not None:
                m.load_state_dict(stopper.best_state)
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
    # Three folds, three jobs, and they must not be confused:
    #   train  -- what the generator fits
    #   val    -- what early stopping selects the model on (never reported)
    #   test   -- what the in-run scoring reports (never trained or selected on)
    # Before early stopping existed, scoring ran on val to keep test pristine.
    # Now that val IS the selection signal, reporting on it would quote a number
    # the model was tuned against, so EVAL_FOLD moved to test. Pass
    # --fold test to test1_prevalence.py / test2_rare_efficacy.py to match.
    EVAL_FOLD = cfg["eval_fold"]
    cache = cfg["cohort_cache"]
    manifest = load_manifest(cache)
    split_sha = manifest_sha256(manifest)
    cohort = list(manifest["hospitals"])
    print(f"Cohort '{manifest.get('cohort_name')}' from {cache}: {cohort}")
    print(f"  split {manifest['fracs']}, guaranteed "
          f"{manifest['guaranteed_folds']}, rare <= "
          f"{manifest['rare_prevalence_max']} per hospital")

    folds = tuple(dict.fromkeys(("train", "val", EVAL_FOLD)))
    clients_by_fold = load_clients(cache, folds=folds)
    clients = {hid: f["train"] for hid, f in clients_by_fold.items()}
    client_tests = {hid: f[EVAL_FOLD] for hid, f in clients_by_fold.items()}
    client_vals = {hid: f["val"] for hid, f in clients_by_fold.items()}
    pooled_train = load_fold(cache, "train")
    pooled_test = load_fold(cache, EVAL_FOLD)
    pooled_val = load_fold(cache, "val")
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
    es_cfg = ({"patience": cfg["es_patience"], "min_delta": cfg["es_min_delta"],
               "min_steps": cfg["es_min_steps"],
               "min_val_patients": cfg["es_min_val_patients"],
               "fallback": cfg["es_fallback"]}
              if cfg["early_stop"] else None)
    if es_cfg:
        eligible = [h for h in cohort
                    if len(client_vals[h]) >= cfg["es_min_val_patients"]]
        print(f"\nEarly stopping: patience {cfg['es_patience']}, min "
              f"{cfg['es_min_steps']} steps, selecting on val, reporting on "
              f"{EVAL_FOLD}")
        print(f"  {len(eligible)}/{len(cohort)} hospitals have >= "
              f"{cfg['es_min_val_patients']} val patients: {eligible}")
        small = [f"{h}({len(client_vals[h])})" for h in cohort
                 if h not in eligible]
        if small:
            print(f"  fallback '{cfg['es_fallback']}' for: {', '.join(small)}")
    else:
        print("\nEarly stopping: OFF -- every regime runs its fixed budget")

    # client_synth[hid] = the synthetic set scored against hospital `hid` in the
    # per-client eval (STEP 7). For fedavg/centralized that's the single global
    # model's output (shared by all); for local it's that hospital's OWN model.
    # Rare-upweighting, built ONCE from the train fold and shared by every
    # regime, so the four arms differ only in how they train -- not in what
    # they consider rare. Keyed on patient_id, which every regime's batches
    # carry, so the pooled centralized arm weights each patient by their own
    # hospital's view exactly as the federated clients do.
    sample_weight_fn = None
    if cfg["rare_upweight"]:
        from utils.rare_weights import make_weight_fn, patient_weights
        print("\nRare-upweighting: w = 1 + sum(rare_threshold / "
              "local_prevalence) over each patient's\n  own-site rare codes, "
              "capped at 20x the no-rare-code baseline, normalised to mean 1 "
              "per site")
        pw = patient_weights(cache, fold="train")
        sample_weight_fn = make_weight_fn(pw, device=device)
        print(f"  {len(pw)} patients weighted")
    else:
        print("\nRare-upweighting: OFF -- every patient counts equally")

    client_synth: Dict[str, List[Dict]] = {}
    try:
        if regime == "local":
            print(f"\nRegime: local-only -- {len(clients)} independent models, "
                  f"{total_epochs} epochs each")
            models = train_local(lambda: build_model(total_epochs), clients,
                                 device=device, writer=writer,
                                 total_epochs=total_epochs, ckpt_dir=save_dir,
                                 ckpt_every=cfg["ckpt_every"],
                                 resume=cfg["resume"], fingerprint=fingerprint,
                                 val_clients=client_vals, es_cfg=es_cfg,
                                 sample_weight_fn=sample_weight_fn,
                                 irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"])
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
                    fingerprint=fingerprint,
                    snapshot_every=cfg["snapshot_every"],
                    val_dataset=pooled_val if es_cfg else None,
                    stopper=(EarlyStopper(cfg["es_patience"],
                                          cfg["es_min_delta"],
                                          cfg["es_min_steps"])
                             if es_cfg else None),
                    sample_weight_fn=sample_weight_fn,
                    irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"])
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
                           split_sha=split_sha,
                           # The federated global is validated on the POOLED
                           # val fold: it is one model serving every site, so
                           # the question it answers is a cohort-wide one, and
                           # 1,217 patients make a far steadier signal than any
                           # single hospital's slice.
                           val_dataset=pooled_val if es_cfg else None,
                           stopper=(EarlyStopper(cfg["es_patience"],
                                                 cfg["es_min_delta"],
                                                 cfg["es_min_steps"])
                                    if es_cfg else None),
                           sample_weight_fn=sample_weight_fn,
                           irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"])
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
                    ckpt_dir=save_dir, fingerprint=fingerprint,
                    val_clients=client_vals, es_cfg=es_cfg,
                    sample_weight_fn=sample_weight_fn,
                    irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"])
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

    # Say plainly whether the booked GPU earned its keep. Cheap to print,
    # and it is the number nobody thinks to look up in sacct afterwards.
    print_gpu_summary()
 