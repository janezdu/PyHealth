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
import math
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from utils.cohort import (
    DEFAULT_CACHE_DIR,
    _to_sample_dataset,
    mix_synthetic,
    read_trajectories,
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
from utils.dpaccount import sgm_epsilon
from utils.dpfed import adapt_clip, dp_aggregate, effective_sigma, sgm_aggregate
from utils.fedparts import (
    describe as describe_parts,
    merge_state as merge_parts,
    personal_keys,
    split_state as split_parts,
)
from generate import proportional_pool, write_synthetic
from eval import evaluate_and_save
from pyhealth.models import HALO
from pyhealth.models.generators.fedreg import clear_prox_ref, set_prox_ref

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
                   choices=["fedavg", "fedavg_ft", "centralized", "local",
                            "fedprox", "ditto", "perada", "perada_fl",
                            "fedrep", "lg_fedavg"],
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
    p.add_argument("--init-from", default=None,
                   help="warm-start every model from another run's trained "
                        "weights: a save dir (its centralized_state.pt or "
                        "fedavg_state.pt) or a .pt file. Unlike --resume this "
                        "starts a NEW run -- round 0, its own save_dir -- and the "
                        "source may come from a different cohort, e.g. a "
                        "public-hospital checkpoint fine-tuned on the private "
                        "sites. Refused unless the source's vocabulary and "
                        "architecture match this run's. run_name gets a "
                        "_from<cohort>R<rounds>E<epochs> suffix.")
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
    p.add_argument("--adapter", default=None,
                   choices=["none", "lora_attn", "last_mlp", "lora_head",
                            "last_mlp_l1", "last_mlp_iht"],
                   help="parameter-efficient LOCAL fine-tuning for fedavg_ft. "
                        "'none' (default) is the existing full fine-tune: all "
                        "6.17M weights per hospital. 'lora_attn' trains LoRA on "
                        "q,v of every attention block (33K, 0.5%%); 'last_mlp' "
                        "trains the final block's feedforward in full (524K, "
                        "8.5%%); 'lora_head' trains LoRA on the autoregressive "
                        "code head (38K, 0.6%%), masked so it cannot break "
                        "within-visit causality. The trunk is frozen either way, "
                        "so the federated signal survives. Only affects the "
                        "fine-tuning stage -- the FedAvg rounds before it, and "
                        "any IRM penalty on them, are untouched. With "
                        "--init-from and --regime centralized/local/fedavg it "
                        "instead freezes the warm-started trunk for the WHOLE "
                        "run; under fedavg only the adapter is then averaged, "
                        "i.e. only the adapter is communicated.")
    p.add_argument("--adapter-rank", type=int, default=None,
                   help="LoRA rank (default 8). Ignored by --adapter last_mlp.")
    p.add_argument("--adapter-mu", type=float, default=None,
                   help="FedProx-style L2 on the adapter parameters, applied "
                        "during fine-tuning only. With a frozen trunk an adapter "
                        "of zero IS the federated global, so this is a proximal "
                        "term: mu controls how far a site may personalise away "
                        "from the shared model. 0 (default) is off.")
    p.add_argument("--adapter-l1", type=float, default=None,
                   help="L1 weight for --adapter last_mlp_l1. Applied as a "
                        "proximal soft-threshold of size lr*lambda after each "
                        "step, NOT as a term in the loss -- a subgradient L1 "
                        "shrinks coordinates without ever reaching zero, so it "
                        "would regularise while reporting 0%% sparsity. Sparsity "
                        "is measured on the delta W - W_global, so the site "
                        "changes few COORDINATES and is the federated global "
                        "everywhere else. 0 (default) is off.")
    p.add_argument("--adapter-sparsity", type=float, default=None,
                   help="Target density for --adapter last_mlp_iht, e.g. 0.1 "
                        "keeps the largest 10%% of delta coordinates (ranked "
                        "globally across the fine-tuned tensors). Projected "
                        "every --adapter-iht-every epochs and on the last "
                        "epoch; training in between is dense, so a pruned "
                        "coordinate can re-grow. 0 (default) is off. Note that "
                        "last_mlp is 525K params, so 0.5 density is still 7x "
                        "lora_head -- around 0.07 is where the parameter counts "
                        "actually meet.")
    p.add_argument("--adapter-iht-every", type=int, default=None,
                   help="Epochs between hard-threshold projections (default 1).")
    p.add_argument("--adapter-optim", default=None, choices=["adam", "sgd"],
                   help="Fine-tuning optimizer (default adam, matching every "
                        "other regime). 'sgd' exists as a correctness "
                        "cross-check for the sparse variants: lr*lambda is the "
                        "EXACT proximal operator under SGD but only a heuristic "
                        "under Adam, and stateless SGD cannot have a pruned "
                        "coordinate re-inflated by a momentum buffer. Pair it "
                        "with --adapter-lr: 1e-4 is an Adam learning rate and "
                        "SGD barely moves at it.")
    p.add_argument("--adapter-lr", type=float, default=None,
                   help="Learning rate for the fine-tuning stage only. "
                        "0/unset uses the model's lr.")
    p.add_argument("--print-save-dir", action="store_true",
                   help="Resolve the config, print the save_dir this run WOULD "
                        "use, and exit without training. A launcher that needs "
                        "the directory before the run exists (to seed a "
                        "checkpoint into it, or to chain scoring onto it) should "
                        "ask for it here rather than rebuilding the name itself "
                        "-- two implementations of the naming rule drift, and "
                        "the failure mode is a run silently scoring against "
                        "another run's directory.")
    p.add_argument("--latent-dim", type=int, default=None,
                   help="width of a per-patient latent z ~ N(0, I), projected "
                        "into the embedding at POSITION 1 -- the conditioning "
                        "slot HALO reserves for a label and this port leaves "
                        "empty. 0 (default) disables it and the model is "
                        "unchanged. That position is chosen because the fine "
                        "head pairs history[t] with visits[t+1], so position 1's "
                        "hidden state is exactly what predicts visit 1, and 79%% "
                        "of this cohort has only visit 1. There is NO encoder "
                        "and no KL: z is always drawn from the prior, so "
                        "training and generation match by construction. What "
                        "makes the model use z instead of averaging it away is "
                        "--xm-k; alone, a latent is ignorable noise.")
    p.add_argument("--dropout", type=float, default=None,
                   help="dropout probability inside HALO -- on the attention "
                        "weights, the attention and MLP residual branches, and "
                        "the embeddings. 0.0 (DEFAULT) reproduces the model as "
                        "it was before this flag existed, exactly; every result "
                        "on the board was produced at 0.0 and is not comparable "
                        "to a dropout run. Above 0 it is also the only source of "
                        "stochasticity in the forward pass, which is what makes "
                        "--xm-k do anything: without dropout the K candidates "
                        "are bit-identical and best-of-K silently trains on 1/K "
                        "of the gradient.")
    p.add_argument("--selftrain-frac", type=float, default=None,
                   help="self-training: each round, after the server broadcast, "
                        "every client generates this fraction of its own train "
                        "size from the JUST-BROADCAST GLOBAL and trains the "
                        "local step on real + synthetic together. 0 (default) "
                        "is off. The synthetic carries cross-site information "
                        "the site does not have, with no raw record leaving it. "
                        "Regenerated fresh each round and never accumulated -- "
                        "an accumulating pool is the self-consuming regime where "
                        "the distribution collapses and the tails go first. "
                        "Watch selftrain_distinct/*: a monotone fall is collapse "
                        "in progress. fedavg/fedavg_ft only.")
    p.add_argument("--selftrain-start", type=int, default=None,
                   help="first ROUND at which self-training engages (default 0). "
                        "Generating from an untrained global is sampling noise, "
                        "so a later start may be the difference between help and "
                        "harm.")
    p.add_argument("--selftrain-at", type=float, default=None,
                   help="WHERE inside local training to generate, as a fraction "
                        "of local_epochs. 0.0 (default) generates once at the "
                        "start of the round, before any local epoch -- the "
                        "original behaviour, and the only value that takes the "
                        "unsplit code path. 0.8 with --local-epochs 10 means 8 "
                        "real-only epochs, then generate, then 2 epochs on the "
                        "mixture. A FRACTION rather than an epoch index so the "
                        "same value means the same thing at a different "
                        "--local-epochs; the resolved integer is what goes in "
                        "the run name. Only meaningful with --selftrain-frac > 0.")
    p.add_argument("--selftrain-source", default=None,
                   choices=["global", "local"],
                   help="WHOSE distribution the synthetic comes from. 'global' "
                        "(default) samples the weights the server just "
                        "broadcast, so a small site mixes in information from "
                        "the other seven -- federated distillation, and the "
                        "only reason to expect more than regularisation. "
                        "'local' samples the partly-adapted local model, which "
                        "carries no information the site does not already have. "
                        "At --selftrain-at 0 the two are identical (nothing has "
                        "trained yet); the gap between them GROWS with "
                        "--selftrain-at, which is what makes the pair the "
                        "experiment rather than a knob.")
    p.add_argument("--xm-k", type=int, default=None,
                   help="best-of-K exploration (Forward XM). 1 (default) is "
                        "ordinary training. Above 1, each batch is run K times "
                        "-- dropout draws a fresh mask each pass -- and only the "
                        "lowest loss PER PATIENT is trained on, so the model has "
                        "to be right once rather than hedge across every "
                        "plausible code. Aimed at the code-count blowup (the "
                        "centralized arm emits 25.2 codes/visit against a real "
                        "11.5) and at rare codes, which MLE smooths away. Costs "
                        "K forward passes and K x activation memory; not "
                        "combinable with --irm-rho.")
    p.add_argument("--irm-schedule", default=None,
                   choices=["step", "linear", "geom", "cosine", "decay"],
                   help="how rho moves once warmup is over. 'step' (default) "
                        "jumps straight to --irm-rho, matching the paper. "
                        "'geom' ramps geometrically, which for a weight "
                        "spanning two decades is the natural continuation -- a "
                        "linear ramp spends 90%% of training above rho/10. "
                        "'linear' and 'cosine' are the other ramps. 'decay' is "
                        "the REVERSE: start at rho and anneal down to 1, which "
                        "tests whether the penalty only has to shape the early "
                        "representation. See irm_rho_at.")
    # ---- literature baselines -------------------------------------------- #
    # These four regimes are points on the same line: each controls only HOW FAR
    # a client may sit from the shared model, by magnitude (ditto, fedprox) or
    # by capacity (fedrep, lg_fedavg, perada). They exist to be compared
    # against, not to win.
    p.add_argument("--fedprox-mu", type=float, default=None,
                   help="FedProx (Li et al., MLSys 2020): weight of "
                        "(mu/2)||w - w_broadcast||^2 added to each client's "
                        "LOCAL loss every round, so no client drifts far from "
                        "the server within a round. Not personalization -- "
                        "there is still one global model at the end. Distinct "
                        "from --adapter-mu, which is an L2 toward ZERO applied "
                        "only during fine-tuning.\n\n"
                        "MIND THE UNITS. The penalty is a SUM over ~6.2M "
                        "parameters; the risk it joins is a MEAN over elements "
                        "and settles near 5e-4. A per-parameter drift of only "
                        "0.001 already makes the penalty ~6, so mu=0.01 -- a "
                        "reasonable value in the paper, where losses are O(1) "
                        "-- contributes ~0.03 here and swamps the risk sixty "
                        "times over. The first sweep at mu=0.01 pinned the "
                        "model to the broadcast weights and emitted 17.7 codes "
                        "per visit against a real 12.5. Useful values here are "
                        "around 1e-6.")
    p.add_argument("--ditto-lambda", type=float, default=None,
                   help="Ditto (Li et al., ICML 2021): weight of "
                        "(lambda/2)||v - w*||^2 while fine-tuning each "
                        "hospital's personal model v, where w* is the frozen "
                        "final FedAvg global. lambda=0 is exactly plain "
                        "fedavg_ft; large lambda pins every site to the global. "
                        "NOTE this is the post-hoc form -- the paper interleaves "
                        "the v update with each round -- which is what the "
                        "equal-budget rule here allows.")
    p.add_argument("--kd-lambda", type=float, default=None,
                   help="PerAda (Xie et al., CVPR 2024): weight of the "
                        "knowledge-distillation term pulling a hospital's "
                        "adapted output distribution toward the frozen global "
                        "model's, as per-code binary KL. Use with an --adapter; "
                        "the adapter is the personalization and KD is the "
                        "brake. Cannot be combined with --xm-k > 1 or "
                        "--irm-rho > 0.")
    # PerAda as the paper specifies it (--regime perada_fl), as opposed to the
    # post-hoc KD fine-tune that --regime perada runs. See run_perada.
    p.add_argument("--rate-mu", type=float, default=None,
                   help="Weight of a relative penalty on the batch's emitted "
                        "code volume vs the observed one, "
                        "((sum p - sum y) / sum y)^2. Aimed at the arms that "
                        "over-emit 2-4x (FedRep, LG-FedAvg). Plain risk only: "
                        "refuses --xm-k > 1 and --irm-rho > 0, whose losses "
                        "are computed elsewhere. 0 (default) = off.")
    p.add_argument("--fedrep-alternate", action="store_true", default=None,
                   help="FedRep's own schedule: each round, train the head "
                        "with the global body frozen, then the body with the "
                        "head frozen. Needs --regime fedrep and "
                        "--local-epochs >= 2.")
    p.add_argument("--perada-lambda", type=float, default=None,
                   help="perada_fl: weight of (lambda/2)||v_m - w||^2 pulling "
                        "each hospital's personal adapter toward the current "
                        "global adapter, every round. Default 0.1, matching the "
                        "Ditto setting in the baselines table.")
    p.add_argument("--perada-kd-steps", type=int, default=None,
                   help="perada_fl: server-side distillation steps per round "
                        "(default 10; the paper uses 100). 0 disables KD, "
                        "which leaves FedAvg-on-adapters plus a Ditto-style "
                        "personal adapter.")
    p.add_argument("--perada-kd-pool", type=int, default=None,
                   help="perada_fl: records the server samples ONCE from the "
                        "frozen trunk to distill on (default 1000).")
    p.add_argument("--perada-kd-batch", type=int, default=None,
                   help="perada_fl: batch size of a distillation step "
                        "(default 64).")
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
    p.add_argument("--dp-clip", type=float, default=None,
                   help="DP-FedAvg L2 clip norm S on each client's UPDATE. "
                        "Enables the DP aggregation path. Set it from the "
                        "measured update norms -- run once with --dp-sigma 0 "
                        "and read the [dp] log lines. Requires "
                        "--weighting uniform.")
    p.add_argument("--dp-sigma", type=float, default=None,
                   help="DP-FedAvg noise multiplier. Gaussian std on the mean "
                        "update is sigma * dp_clip / n_clients. 0 clips but "
                        "adds no noise, which is the control that separates "
                        "the cost of clipping from the cost of noise. "
                        "utils/dpaccount.py converts (sigma, rounds) to "
                        "epsilon; all 8 sites participate every round, so no "
                        "subsampling amplification is claimed.")
    p.add_argument("--dp-mech", choices=["gaussian", "sgm"], default=None,
                   help="Noise mechanism. 'gaussian' (default) is server-side "
                        "DP-FedAvg. 'sgm' is Fed-SGM (Li et al. 2025): each "
                        "client sketches its clipped update and adds noise "
                        "before sending, with --dp-sigma as the ABSOLUTE noise "
                        "std per sketch coordinate. Needs --dp-sketch-b.")
    p.add_argument("--dp-sketch-b", type=int, default=None,
                   help="Sketch dimension b for --dp-mech sgm.")
    p.add_argument("--dp-adaptive", type=float, default=None,
                   help="Target quantile for ADAPTIVE clipping (Andrew et al. "
                        "2021). 0.5 aims the clip at the median client norm. "
                        "Omitted/0 keeps --dp-clip fixed. The probe measured "
                        "norms decaying ~30x over 10 rounds, which no single "
                        "fixed clip fits.")
    p.add_argument("--dp-sigma-b", type=float, default=None,
                   help="Noise multiplier on the adaptive quantile bit. It is "
                        "a second release, so its cost ADDS: use "
                        "dpfed.effective_sigma(sigma, sigma_b) for epsilon. "
                        "sigma_b = 4*sigma costs ~3%%.")
    p.add_argument("--dp-adapt-lr", type=float, default=None,
                   help="Step size for the geometric clip update. Default 1.0, "
                        "NOT the usual 0.2: at 10 rounds a slower rate never "
                        "catches the decay (replay lands at 0.535 vs a true "
                        "median of 0.234).")
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
    if cfg.get("regime") in FT_REGIMES:
        name += f"_ft{cfg.get('ft_epochs', cfg['local_epochs'])}"
    # Baseline knobs. Two lambdas sharing a save_dir would clobber each other's
    # checkpoints and results; see the comment below about warmup for what that
    # costs when it goes unnoticed.
    if cfg.get("fedprox_mu"):
        name += f"_prox{cfg['fedprox_mu']:g}"
    if cfg.get("ditto_lambda"):
        name += f"_ditto{cfg['ditto_lambda']:g}"
    if cfg.get("kd_lambda"):
        name += f"_kd{cfg['kd_lambda']:g}"
    if cfg.get("rate_mu"):
        name += f"_rate{cfg['rate_mu']:g}"
    if cfg.get("fedrep_alternate"):
        name += "_alt"
    if cfg.get("regime") == "perada_fl":
        # All three change the trained arm, so all three are in the name.
        name += (f"_pl{cfg.get('perada_lambda', 0.1):g}"
                 f"_kds{cfg.get('perada_kd_steps', 10)}"
                 f"p{cfg.get('perada_kd_pool', 1000)}")
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
        # WARMUP BELONGS IN THE NAME. Without it, two runs that differ only in
        # when the penalty engages resolve to the SAME save_dir and silently
        # overwrite each other's checkpoints and results -- and the comparison
        # between them becomes a comparison of one run with itself. Caught when
        # a warmup sweep at fixed rho produced one directory for two arms.
        if cfg.get("irm_warmup", 0):
            name += f"w{cfg['irm_warmup']}"
    # Full-budget runs are a different model from early-stopped ones (the
    # hilo8 centralized baseline stopped at 37 of 100 epochs), so they get
    # their own save_dir rather than overwriting the baseline.
    if cfg.get("early_stop") is False:
        name += "_nes"
    if (cfg.get("dp_clip") or 0.0) > 0:
        # Both go in the name. Same reasoning as irm rho+warmup: two DP settings
        # resolving to one save_dir makes a sweep a comparison of a run with
        # itself.
        name += f"_dpc{cfg['dp_clip']:g}s{cfg.get('dp_sigma') or 0.0:g}"
        if (cfg.get("dp_adaptive") or 0.0) > 0:
            name += f"ad{cfg['dp_adaptive']:g}"
        if cfg.get("dp_mech") == "sgm":
            name += f"_sgmb{cfg['dp_sketch_b']}"
    if cfg.get("latent_dim", 0) > 0:
        name += f"_z{cfg['latent_dim']}"
    if cfg.get("dropout", 0.0) > 0:
        name += f"_do{cfg['dropout']:g}"
    if cfg.get("xm_k", 1) > 1:
        name += f"_xm{cfg['xm_k']}"
    if cfg.get("selftrain_frac", 0.0) > 0:
        name += f"_st{cfg['selftrain_frac']:g}"
        if cfg.get("selftrain_start", 0):
            name += f"r{cfg['selftrain_start']}"
        # WHERE the generation happens and WHOSE weights produced it are both
        # different models, so they must not share a save_dir. Appended only
        # when non-default: at at=0 / source=global this is the original
        # behaviour and must keep resolving to the name it already has, or the
        # existing st runs become unfindable.
        at = float(cfg.get("selftrain_at", 0.0) or 0.0)
        if at > 0:
            # The resolved epoch, not the fraction: "@8" is what someone reading
            # a log wants, and it is unambiguous without knowing local_epochs.
            name += f"@{_selftrain_split_epoch(at, cfg['local_epochs'])}"
        if cfg.get("selftrain_source", "global") != "global":
            name += f"_src{cfg['selftrain_source']}"
    # A different rho schedule is a different training trajectory, so it needs
    # its own save_dir even at identical rho and warmup.
    if cfg.get("irm_rho", 0.0) > 0 and cfg.get("irm_schedule", "step") != "step":
        name += f"_{cfg['irm_schedule']}"
    # The adapter changes WHICH weights fine-tuning moves, so an adapter run is
    # a different model from the full fine-tune and needs its own save_dir. mu
    # is in the name too: sweeping it is the point.
    if cfg.get("adapter", "none") != "none":
        name += f"_{cfg['adapter']}r{cfg.get('adapter_rank', 8)}"
        if cfg.get("adapter_mu", 0.0) > 0:
            name += f"_mu{cfg['adapter_mu']:g}"
        # Two runs of the same variant at different lambda / density are
        # different models. Without these in the name the second silently
        # overwrites the first's save_dir, checkpoints and results.
        if cfg.get("adapter_l1", 0.0) > 0:
            name += f"_l1{cfg['adapter_l1']:g}"
        if cfg.get("adapter_sparsity", 0.0) > 0:
            name += f"_sp{cfg['adapter_sparsity']:g}"
            if cfg.get("adapter_iht_every", 1) != 1:
                name += f"e{cfg['adapter_iht_every']}"
        if cfg.get("adapter_optim", "adam") != "adam":
            name += f"_{cfg['adapter_optim']}"
        if cfg.get("adapter_lr", 0.0):
            name += f"_alr{cfg['adapter_lr']:g}"
    # A warm-started run is a different model from the same arm trained from
    # scratch, and two sources (pub8 at 25 vs 100 rounds) are different models
    # from each other, so the source's cohort and budget both go in the name.
    if cfg.get("init_from"):
        _, src = _init_source(cfg["init_from"])
        src_cohort = os.path.basename(str(src["cohort_cache"]).rstrip("/"))
        name += f"_from{src_cohort}R{src['n_rounds']}E{src['local_epochs']}"
    return name


#: Architecture knobs a warm start must agree on. Matching tensor shapes is not
#: enough: n_heads changes how attention splits a weight without changing its
#: shape, so a mismatch would load cleanly and mean nothing.
INIT_ARCH_KEYS = ("embed_dim", "n_heads", "n_layers", "n_ctx", "latent_dim")

#: Checkpoint files ``--init-from`` accepts in a save dir.
INIT_CKPTS = ("centralized_state.pt", "fedavg_state.pt")


def _init_source(path: str) -> Tuple[str, dict]:
    """Resolve ``--init-from`` to ``(checkpoint file, source run's config)``.

    The source's ``config.json`` is required, not optional: it is the only
    record of which vocabulary and architecture the weights were trained for,
    and without it neither can be checked.
    """
    if os.path.isdir(path):
        found = [os.path.join(path, f) for f in INIT_CKPTS
                 if os.path.exists(os.path.join(path, f))]
        if len(found) != 1:
            raise FileNotFoundError(
                f"--init-from {path}: expected exactly one of {INIT_CKPTS} in "
                f"that directory, found {len(found)}. Pass the .pt file itself.")
        ckpt = found[0]
    elif os.path.exists(path):
        ckpt = path
    else:
        raise FileNotFoundError(f"--init-from {path}: no such file or directory")
    cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"--init-from {ckpt}: no config.json beside it, so its vocabulary "
            f"and architecture cannot be verified. Refusing to warm-start.")
    with open(cfg_path) as fh:
        return ckpt, json.load(fh)


def load_init_state(path: str, cfg: dict, code_vocab: Dict[str, int],
                    log: Callable[[str], None] = print
                    ) -> Dict[str, torch.Tensor]:
    """Load ``--init-from`` weights after proving they fit this run.

    Two checks, both of which would otherwise fail silently:

    * **Vocabulary.** The embedding rows and output columns are indexed by
      code. Two caches with the same 921 codes in a different order produce
      identically shaped tensors, so the load succeeds and every code is read
      as some other code. The code->index maps are compared exactly; the
      ``vocab.json`` files themselves can differ in legacy fields that
      ``load_processor`` ignores, so comparing bytes would be a false alarm.
    * **Architecture.** See :data:`INIT_ARCH_KEYS`.

    Returns:
        The source's ``global_state``.
    """
    from utils.cohort import load_processor

    ckpt, src = _init_source(path)
    bad = {k: (src.get(k), cfg.get(k)) for k in INIT_ARCH_KEYS
           if src.get(k) != cfg.get(k)}
    if bad:
        raise ValueError(
            f"--init-from {ckpt} was trained with a different architecture: "
            + ", ".join(f"{k} source={a} here={b}" for k, (a, b) in bad.items()))
    src_vocab = load_processor(src["cohort_cache"]).code_vocab
    if src_vocab != code_vocab:
        moved = sum(1 for c, i in code_vocab.items() if src_vocab.get(c) != i)
        raise ValueError(
            f"--init-from {ckpt} was trained on {src['cohort_cache']}, whose "
            f"vocabulary differs from this cohort's ({len(src_vocab)} vs "
            f"{len(code_vocab)} codes, {moved} at a different index). The "
            f"weights would load and decode every code as the wrong one.")
    payload = torch.load(ckpt, map_location="cpu")
    log(f"Init: warm start from {ckpt} (cohort "
        f"{os.path.basename(str(src['cohort_cache']).rstrip('/'))}, "
        f"{src['regime']} R{src['n_rounds']} E{src['local_epochs']}); "
        f"vocabulary matches ({len(code_vocab)} codes)")
    return payload["global_state"]


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
        "resume": args.resume, "init_from": args.init_from,
        "cohort_cache": args.cohort_cache,
        "ckpt_every": args.ckpt_every,
        "snapshot_every": args.snapshot_every,
        "rare_upweight": args.rare_upweight,
        "dp_clip": args.dp_clip, "dp_sigma": args.dp_sigma,
        "dp_mech": args.dp_mech, "dp_sketch_b": args.dp_sketch_b,
        "dp_adaptive": args.dp_adaptive, "dp_sigma_b": args.dp_sigma_b,
        "dp_adapt_lr": args.dp_adapt_lr,
        "tb_logdir": args.tb_logdir, "no_tb": args.no_tb,
        "log_every_epochs": args.log_every_epochs,
        "n_rounds": args.n_rounds, "local_epochs": args.local_epochs,
        "num_synth": args.num_synth, "metrics": args.metrics,
        "synth_per_hospital": args.synth_per_hospital,
        "ft_epochs": args.ft_epochs, "run_name": args.run_name,
        "fedprox_mu": args.fedprox_mu, "ditto_lambda": args.ditto_lambda,
        "kd_lambda": args.kd_lambda,
        "perada_lambda": args.perada_lambda,
        "rate_mu": args.rate_mu,
        "fedrep_alternate": args.fedrep_alternate,
        "perada_kd_steps": args.perada_kd_steps,
        "perada_kd_pool": args.perada_kd_pool,
        "perada_kd_batch": args.perada_kd_batch,
        "irm_rho": args.irm_rho, "irm_warmup": args.irm_warmup,
        "irm_schedule": args.irm_schedule, "xm_k": args.xm_k,
        "dropout": args.dropout, "latent_dim": args.latent_dim,
        "selftrain_frac": args.selftrain_frac,
        "selftrain_start": args.selftrain_start,
        "selftrain_at": args.selftrain_at,
        "selftrain_source": args.selftrain_source,
        "adapter": args.adapter, "adapter_rank": args.adapter_rank,
        "adapter_mu": args.adapter_mu, "adapter_l1": args.adapter_l1,
        "adapter_sparsity": args.adapter_sparsity,
        "adapter_iht_every": args.adapter_iht_every,
        "adapter_optim": args.adapter_optim, "adapter_lr": args.adapter_lr,
        "print_save_dir": args.print_save_dir or None,
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
    cfg.setdefault("dp_clip", 0.0)
    cfg.setdefault("dp_sigma", 0.0)
    cfg.setdefault("dp_mech", "gaussian")
    cfg.setdefault("dp_sketch_b", 0)
    cfg.setdefault("dp_adaptive", 0.0)
    cfg.setdefault("dp_sigma_b", 0.0)
    cfg.setdefault("dp_adapt_lr", 1.0)
    cfg.setdefault("resume", False)
    cfg.setdefault("init_from", None)
    cfg.setdefault("cohort_cache", DEFAULT_CACHE_DIR)
    cfg.setdefault("ckpt_every", 1)
    cfg.setdefault("snapshot_every", 0)      # 0 = keep no numbered snapshots
    cfg.setdefault("rare_upweight", False)
    cfg.setdefault("adapter", "none")
    cfg.setdefault("adapter_rank", 8)
    cfg.setdefault("adapter_mu", 0.0)
    cfg.setdefault("adapter_l1", 0.0)
    cfg.setdefault("adapter_sparsity", 0.0)
    cfg.setdefault("adapter_iht_every", 1)
    cfg.setdefault("adapter_optim", "adam")
    cfg.setdefault("adapter_lr", 0.0)
    cfg.setdefault("fedprox_mu", 0.0)
    cfg.setdefault("ditto_lambda", 0.0)
    cfg.setdefault("kd_lambda", 0.0)
    cfg.setdefault("perada_lambda", 0.1)
    cfg.setdefault("rate_mu", 0.0)
    cfg.setdefault("fedrep_alternate", False)
    cfg.setdefault("perada_kd_steps", 10)
    cfg.setdefault("perada_kd_pool", 1000)
    cfg.setdefault("perada_kd_batch", 64)
    cfg.setdefault("irm_rho", 0.0)
    cfg.setdefault("irm_warmup", 0)
    cfg.setdefault("irm_schedule", "step")
    cfg.setdefault("xm_k", 1)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("latent_dim", 0)
    cfg.setdefault("selftrain_frac", 0.0)
    cfg.setdefault("selftrain_start", 0)
    cfg.setdefault("selftrain_at", 0.0)
    cfg.setdefault("selftrain_source", "global")
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
               key: str = "completed_rounds", extra: dict = None):
    """Atomically write the FedAvg checkpoint (tmp file + rename).

    The tmp-then-rename keeps the checkpoint valid even if the job is killed
    mid-write -- ``os.replace`` is atomic on POSIX, so we never leave a
    half-written file that a later resume would choke on.

    Args:
        extra: Additional top-level entries. Used by the key-partitioned
            regimes to carry their per-client personal weights, which are NOT
            in ``global_state`` and would otherwise be lost on resume -- and
            lost silently, since the run would simply continue as plain FedAvg.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    payload = {key: completed, "global_state": global_state,
               "fingerprint": fingerprint}
    payload.update(extra or {})
    torch.save(payload, tmp)
    os.replace(tmp, path)


#: Regimes that spend part of the epoch budget on per-hospital fine-tuning
#: rather than on federated rounds. They all share fedavg_ft's shape: FedAvg
#: first, then a personal model per site. Listing them here is what keeps the
#: budget split (total passes/hospital == total_epochs) equal across arms.
FT_REGIMES = ("fedavg_ft", "ditto", "perada", "perada_fl")

#: Regimes that run FedAvg rounds and produce ONE global generator.
FEDAVG_REGIMES = ("fedavg", "fedavg_ft", "fedprox", "ditto", "perada",
                  "perada_fl",
                  "fedrep", "lg_fedavg")

#: Which parameters each key-partitioned regime keeps at the client.
PERSONAL_MODE = {"fedrep": "head", "lg_fedavg": "body"}

#: Regimes where ONE generator serves every hospital. Everything else emits a
#: per-hospital set, which is what `shared_generator` in the written synthetic
#: tells the scoring scripts -- get this wrong and eight personalized models
#: collapse into one classifier in Test 2.
SHARED_GENERATOR_REGIMES = ("centralized", "fedavg", "fedprox")

#: Accepted ``--irm-schedule`` values; see :func:`irm_rho_at`.
IRM_SCHEDULES = ("step", "linear", "geom", "cosine", "decay")


def irm_rho_at(step: int, rho: float, warmup: int,
               schedule: str = "step", total: int = 0) -> float:
    """IRMv1's penalty weight for one round/epoch, with the paper's warmup.

    IRM cannot be trained by simply switching a large penalty on: at a big
    ``rho`` the penalty dominates from step zero, the model finds the trivial
    invariant solution (predict the same thing everywhere) and never fits the
    data at all. The published recipe trains at ``rho ~ 1`` for a warmup period
    -- long enough to reach a useful representation -- then jumps to the large
    value. A run that "shows IRM does nothing" without warmup usually never
    fitted in the first place.

    ``schedule="step"`` is the default and matches the reference
    implementation: a hard jump, not a ramp. A ramp conflates "the penalty grew"
    with "training progressed" and makes the loss curve harder to read at
    exactly the point you need it. The alternatives exist because the step is
    also the crudest possible choice, and this cohort has evidence the timing
    matters -- starting the penalty late (epoch 68) recovered less than half the
    gap that starting at 34 did, which says the penalty PREVENTS the failure
    rather than repairing it.

    Args:
        step: Completed EPOCHS. FedAvg converts its round counter at the call
            site (round * local_epochs) so one --irm-warmup value means the same
            point in training for every regime. Before this conversion existed,
            --irm-warmup 17 engaged fedavg at epoch 34 and centralized at epoch
            17, and the two arms were not comparable on when the penalty began.
        rho: The post-warmup penalty weight.
        warmup: Steps to hold at 1.0 first. ``0`` applies ``rho`` immediately.
        schedule: How the weight moves once warmup is over.

            * ``"step"`` -- jump straight to ``rho`` (default, the paper's).
            * ``"linear"`` -- ramp 1 -> rho linearly over the remaining epochs.
            * ``"geom"`` -- ramp 1 -> rho GEOMETRICALLY (``rho ** t``). For a
              penalty weight spanning two decades this is the natural
              continuation: it spends equal time per order of magnitude,
              where a linear ramp spends 90% of training above rho/10.
            * ``"cosine"`` -- a smooth 1 -> rho ramp, slow at both ends.
            * ``"decay"`` -- the REVERSE: start at ``rho`` and anneal down to 1.
              Tests whether the penalty's job is to shape the early
              representation and then get out of the way, which is the opposite
              of the paper's reading and worth falsifying directly.

        total: Total epochs in the run, needed by every schedule except
            ``"step"`` to know how far along it is. Ignored by ``"step"``; a
            ramp with ``total <= warmup`` degrades to ``"step"`` rather than
            dividing by zero.

    Returns:
        The penalty weight to pass to ``train_model(irm_rho=...)``.

    Raises:
        ValueError: On an unknown schedule name.
    """
    if rho <= 0.0:
        return 0.0
    if schedule not in IRM_SCHEDULES:
        raise ValueError(f"unknown --irm-schedule {schedule!r}; choose from "
                         f"{list(IRM_SCHEDULES)}")
    if schedule == "step":
        return 1.0 if step < warmup else rho
    if step < warmup:
        # Every ramp still honours the warmup: the model has to fit something
        # before the penalty is allowed to matter at all.
        return 1.0
    span = total - warmup
    if span <= 0:
        return rho
    t = min(1.0, max(0.0, (step - warmup) / span))
    if schedule == "linear":
        return 1.0 + (rho - 1.0) * t
    if schedule == "geom":
        return float(rho ** t)
    if schedule == "cosine":
        return 1.0 + (rho - 1.0) * (1.0 - math.cos(math.pi * t)) / 2.0
    # "decay": start hot and cool down -- rho at the warmup boundary, 1 at the end.
    return float(rho ** (1.0 - t))


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


def _selftrain_split_epoch(at: float, local_epochs: int) -> int:
    """Resolve ``--selftrain-at`` (a fraction) to an epoch index.

    Returns the number of real-only epochs to run before generating. Clamped to
    ``[0, local_epochs - 1]``: a split at the very end would generate synthetic
    data and then never train on it, which looks like a valid run and silently
    measures nothing. 0 means "no split" and takes the original code path.

    Args:
        at: Fraction of local epochs to run before generating.
        local_epochs: Local epochs per round.

    Returns:
        Number of real-only epochs before the generation point.
    """
    if not at or at <= 0:
        return 0
    return max(0, min(int(round(at * local_epochs)), local_epochs - 1))


def _synth_health(synthetic: List[dict]) -> dict:
    """Distinct codes and codes-per-visit in one generated batch.

    The tail-collapse detector for self-training. Feeding a generator its own
    output narrows the distribution and the TAILS GO FIRST, which on this cohort
    is precisely what is being measured -- so the failure has to be watched
    directly rather than inferred from a prevalence score at the end of the run.
    A monotone fall in ``distinct`` across rounds is the signal to stop; it shows
    up well before Test 1 moves, and a single end-of-run number cannot tell
    "stable and better" from "two rounds short of collapse".
    """
    seen, n_codes, n_visits = set(), 0, 0
    for p in synthetic:
        for visit in p.get("visits", []):
            n_visits += 1
            n_codes += len(visit)
            seen.update(visit)
    return {"distinct": len(seen),
            "codes_per_visit": n_codes / n_visits if n_visits else 0.0,
            "n_visits": n_visits}


def log_xm(writer, model, step: int, tag: str) -> None:
    """Write best-of-K's two health signals as their own curves.

    Neither is visible in the training loss, and both mark the same degenerate
    state from opposite sides: if dropout does not separate the K candidates
    then ``spread`` goes to zero, ``win_entropy`` stops being uniform, and the
    run is quietly training on 1/K of its gradient while the loss curve looks
    entirely normal.

    ``win_entropy`` is normalised to [0, 1]: 1.0 means every candidate wins
    equally often, which is what a working exploration looks like.

    No-ops when the run is not using XM.
    """
    if writer is None:
        return
    xm = getattr(model, "last_xm", None)
    if not xm:
        return
    writer.add_scalar(f"xm_spread/{tag}", xm["spread"], step)
    writer.add_scalar(f"xm_win_entropy/{tag}", xm["win_entropy"], step)


def log_sparse(writer, model, step: int, tag: str) -> None:
    """Write the L1 norm and non-zero fraction of the fine-tuning delta.

    Both, not one. ``nnz_frac`` alone cannot tell a genuinely sparse update from
    one that kept 7% of coordinates and put an enormous magnitude in each;
    ``l1`` alone cannot tell shrinkage from sparsity, which is exactly the
    failure a subgradient L1 produces -- every coordinate small, none of them
    zero. The pair is what makes the claim checkable, and neither is visible in
    the training loss: a dense run and a 7%-dense one can sit at the same loss.

    No-ops when the run is not using a delta-sparse variant, so every regime can
    call it blind.
    """
    if writer is None:
        return
    st = getattr(model, "last_sparse", None)
    if not st:
        return
    writer.add_scalar(f"sparse_l1/{tag}", st["l1"], step)
    writer.add_scalar(f"sparse_l2/{tag}", st["l2"], step)
    writer.add_scalar(f"sparse_nnz_frac/{tag}", st["nnz_frac"], step)
    writer.add_scalar(f"sparse_linf/{tag}", st["linf"], step)


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
    irm_schedule: str = "step",
    xm_k: int = 1,
    selftrain_frac: float = 0.0,
    selftrain_start: int = 0,
    selftrain_at: float = 0.0,
    selftrain_source: str = "global",
    client_trajectories: Dict[str, dict] = None,
    processor=None,
    personal_mode: str = "none",
    prox_mu: float = 0.0,
    dp_clip: float = 0.0,
    dp_sigma: float = 0.0,
    dp_adaptive: float = 0.0,
    dp_sigma_b: float = 0.0,
    dp_adapt_lr: float = 1.0,
    rate_mu: float = 0.0,
    fedrep_alternate: bool = False,
    adapter_lr: float = 0.0,
    dp_mech: str = "gaussian",
    dp_sketch_b: int = 0,
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
    # Weights each client keeps out of the average: {cid: {key: tensor}}. Empty
    # for plain FedAvg, and empty on round 0 for everyone, so every client still
    # starts from identical broadcast weights -- which FedRep and LG-FedAvg both
    # require.
    personal: Dict[str, Dict[str, torch.Tensor]] = {}
    pkeys = personal_keys(global_state, personal_mode)
    if personal_mode != "none":
        log(f"FedAvg: personal_mode={personal_mode} keeps "
            f"{describe_parts(global_state, pkeys)} at each client")
    # A frozen trunk (--adapter with --init-from) never moves, so it is never
    # sent: only the trainable keys are averaged and the rest stay at the
    # broadcast value. Averaging them anyway is a no-op in exact arithmetic,
    # but it would misstate what the federation communicates. None means
    # every key is shared, which keeps plain FedAvg on its original path.
    comm = None
    if any(not p.requires_grad for p in model.parameters()):
        comm = {n for n, p in model.named_parameters() if p.requires_grad}
        if pkeys or (dp_clip > 0 and dp_mech != "sgm") or fedrep_alternate:
            raise ValueError(
                "a frozen trunk cannot be combined with a personal slice, "
                "Gaussian DP-FedAvg or fedrep_alternate: each assumes it owns "
                "the choice of which keys are trained and averaged.")
        n_comm = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        log(f"FedAvg: trunk frozen -- each client sends {n_comm:,} of "
            f"{n_all:,} params per round ({100 * n_comm / n_all:.2f}%)")
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
        if personal_mode != "none":
            # Resuming without these would quietly continue as plain FedAvg --
            # no error, plausible numbers, wrong method. Refuse instead.
            if "personal_states" not in ckpt:
                raise ValueError(
                    f"{ckpt_path} has no 'personal_states' but "
                    f"personal_mode={personal_mode!r} needs them. It was "
                    f"written by a plain-FedAvg run; resuming would silently "
                    f"drop the per-client weights. Delete it and start fresh."
                )
            personal = ckpt["personal_states"]
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
    split_epoch = _selftrain_split_epoch(selftrain_at, local_epochs)
    # FedRep's own schedule (Collins et al. 2021): each round the client first
    # fits its HEAD with the received global body frozen, then updates the BODY
    # with its head frozen. Joint training -- the default here -- lets the head
    # track a body that is about to be averaged away, which is the suspected
    # cause of FedRep's over-emission: the released model pairs the final
    # averaged body with a head fitted to a different one.
    if fedrep_alternate:
        if personal_mode != "head":
            raise ValueError("fedrep_alternate is FedRep's schedule; it needs "
                             f"personal_mode='head', got {personal_mode!r}")
        if local_epochs < 2:
            raise ValueError("fedrep_alternate splits each round into a head "
                             "block and a body block, so it needs "
                             "local_epochs >= 2")
        if selftrain_frac > 0:
            raise ValueError("fedrep_alternate cannot be combined with "
                             "self-training")
    if selftrain_frac > 0:
        log(f"self-training: frac={selftrain_frac:g} from round {selftrain_start}, "
            f"generating after epoch {split_epoch}/{local_epochs} from the "
            f"{selftrain_source} model")

    def _generate_and_mix(model, cid, r, at_epoch, source, global_state):
        """Sample synthetic patients for one client and mix them with its real fold.

        Args:
            model: The client's model, mid-local-training when ``at_epoch > 0``.
            cid: Hospital id.
            r: Round index, for logging and the TensorBoard step.
            at_epoch: Local epoch this generation happens after. 0 is the
                round-start generation.
            source: ``"global"`` samples the weights the server broadcast;
                ``"local"`` samples the model as it currently stands. At
                ``at_epoch == 0`` these are the same weights.
            global_state: The broadcast state, needed for ``source="global"``
                once local training has already moved the weights.

        Returns:
            A ``SampleDataset`` over real + synthetic, or ``None`` when the
            requested count rounds to zero -- the caller then keeps the real
            dataset it already holds rather than rebuilding an identical one.
        """
        real = client_trajectories[cid]
        n_syn = int(round(selftrain_frac * sizes[cid]))
        if n_syn <= 0:
            return None

        # Sampling the GLOBAL after local training has begun means temporarily
        # putting the broadcast weights back. The local weights are restored
        # before training resumes -- without that, the second block would
        # continue from the global and the first block's epochs would be thrown
        # away silently.
        restore = None
        if source == "global" and at_epoch > 0:
            restore = _snapshot(model)
            model.load_state_dict(global_state)
        try:
            syn = model.generate(n_syn, device=device)
        finally:
            if restore is not None:
                model.load_state_dict(restore)

        mixed, kept = mix_synthetic(real, syn, cid, processor,
                                    f"mix_{cid}_r{r}e{at_epoch}")
        st = _synth_health(syn)
        log(f"  round {r + 1} client {cid} @e{at_epoch} ({source}): "
            f"+{kept}/{n_syn} synthetic "
            f"({100 * kept / max(1, n_syn):.0f}% non-empty), "
            f"{st['distinct']} distinct codes, "
            f"{st['codes_per_visit']:.2f} codes/visit")
        if writer is not None:
            # Stepped by the epoch the generation actually happened at, not by
            # the round, so a split run's scalars land where the model that
            # produced them lived.
            step = r * local_epochs + at_epoch
            # Tail collapse shows here GENERATIONS before it reaches
            # any prevalence metric: a monotone fall in the distinct
            # code count is the stop signal.
            writer.add_scalar(f"selftrain_distinct/hospital_{cid}",
                              st["distinct"], step)
            writer.add_scalar(f"selftrain_codes_per_visit/hospital_{cid}",
                              st["codes_per_visit"], step)
            writer.add_scalar(f"selftrain_kept/hospital_{cid}",
                              kept / max(1, n_syn), step)
        return mixed
    # The trained parameters: the only keys DP may noise. See dp_aggregate.
    dp_keys = comm if comm is not None else {
        n for n, p in model.named_parameters() if p.requires_grad}
    if dp_mech == "sgm":
        if dp_clip <= 0 or dp_sigma <= 0 or dp_sketch_b <= 0:
            raise ValueError("--dp-mech sgm needs --dp-clip, --dp-sigma and "
                             "--dp-sketch-b, all positive")
        if dp_adaptive > 0:
            raise ValueError("adaptive clipping's quantile release is not in "
                             "the Fed-SGM accounting")
        log(f"[dp] Fed-SGM: {len(dp_keys)} tensors, b={dp_sketch_b}, "
            f"clip={dp_clip:g}, sigma_g={dp_sigma:g} -> epsilon "
            f"{sgm_epsilon(dp_sigma, dp_sketch_b, n_rounds, dp_clip)[0]:.2f}")
    elif dp_clip > 0 and dp_sigma > 0:
        # The adaptive bit is a SECOND release about the same data, so its RDP
        # adds. Report the sigma the accountant must use, not the raw one.
        se = effective_sigma(dp_sigma, dp_sigma_b) if dp_adaptive > 0 else dp_sigma
        log(f"[dp] clip={dp_clip:g} sigma={dp_sigma:g}"
            + (f" adaptive(target={dp_adaptive:g}, lr={dp_adapt_lr:g},"
               f" sigma_b={dp_sigma_b:g})" if dp_adaptive > 0 else "")
            + f"  -> EFFECTIVE sigma for epsilon = {se:.4f}")

    if dp_clip > 0 and pkeys:
        raise ValueError(
            "DP-FedAvg is not supported with a personal parameter slice "
            f"(personal_mode={personal_mode!r}). The clipped delta and the "
            "reference state would cover different keys, and the sensitivity "
            "bound would not hold over the half that is never aggregated.")

    # What a frozen-trunk run moved, measured from where it started. An adapter
    # that barely trains and one that trains to no effect score the same, so
    # only this tells them apart.
    comm_start = ({k: global_state[k].clone() for k in comm}
                  if comm is not None else None)

    for r in range(start_round, n_rounds):
        # The state as BROADCAST this round. DP clips each client delta against
        # this, not against the post-aggregation state, so it has to be taken
        # before any client trains and kept for the aggregation step.
        round_ref = ({k: v.detach().clone() for k, v in global_state.items()}
                     if dp_clip > 0 else None)
        snapshots: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        round_final_losses: List[float] = []
        if prox_mu > 0.0:
            # FedProx: penalise drift from THIS round's broadcast weights. Built
            # once per round -- every client is pulled toward the same point,
            # and ProxRef reads the live parameters at penalty() time, so one
            # object serves all eight. Rebuilding it per client cloned 25 MB
            # four hundred times and was most of a 2h15 runtime.
            set_prox_ref(model.halo_model, global_state)
        for cid in client_ids:
            # Broadcast, but keep whatever this client held back last round.
            # A bare load_state_dict(global_state) here is what turns FedRep or
            # LG-FedAvg back into FedAvg without any error.
            model.load_state_dict(merge_parts(global_state, personal.get(cid, {})))

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
                    log_xm(writer, model, step, f"hospital_{cid}")

            # SELF-TRAINING. Generate from the model the server just broadcast,
            # then train the local step on real + synthetic together. The
            # synthetic is sampled from the AGGREGATED GLOBAL, so what a small
            # hospital mixes in carries information from the other seven sites
            # without any raw record leaving anyone's machine -- that, rather
            # than "more data", is the reason to expect anything from it.
            #
            # Fresh every round and never accumulated: a growing pool of an
            # earlier model's output is the self-consuming regime where the
            # distribution collapses, and the tails go first. Rare codes are the
            # whole evaluation here, so the safer arrangement is the one where
            # real data is present at every step and the synthetic is always
            # from the current global.
            local_train = clients[cid]
            # Round-start generation. Skipped entirely when split_epoch > 0:
            # there the whole point is that the first block sees REAL DATA ONLY,
            # and generating here as well would put synthetic in front of it.
            if (split_epoch == 0 and selftrain_frac > 0
                    and r >= selftrain_start):
                local_train = _generate_and_mix(
                    model, cid, r, at_epoch=0, source="global",
                    global_state=global_state) or local_train

            def _train_block(dataset, n_epochs, epoch_offset):
                """Run ``n_epochs`` of local training, numbered from ``epoch_offset``.

                The epoch offset exists so a split round keeps ONE continuous
                epoch axis: the second block's callback still reports epochs
                8, 9 rather than restarting at 0, so TensorBoard curves and the
                round-final-loss capture behave the same split or not.
                """
                prev = model._epochs
                model._epochs = n_epochs
                try:
                    model.train_model(
                        dataset, val_dataset=None, device=device,
                        on_epoch_end=lambda e, ml: _on_epoch_end(
                            e + epoch_offset, ml),
                        sample_weight_fn=sample_weight_fn,
                        # warmup is in EPOCHS everywhere; convert to
                        # rounds here so --irm-warmup means the same
                        # point in training for every regime.
                        irm_rho=irm_rho_at(
                            r * local_epochs + epoch_offset, irm_rho,
                            irm_warmup, irm_schedule,
                            n_rounds * local_epochs),
                        xm_k=xm_k,
                        prox_mu=prox_mu,
                        rate_mu=rate_mu,
                        adapter_lr=adapter_lr)
                finally:
                    model._epochs = prev

            if split_epoch > 0 and selftrain_frac > 0 and r >= selftrain_start:
                # THE OUROBOROS PATH. Train on real data only, then generate
                # from whatever the model has become, then finish the round on
                # the mixture.
                #
                # NOT equivalent to the unsplit path with the same total epochs,
                # even ignoring the synthetic: train_model builds its Adam
                # inside the call, so the second block starts with fresh moment
                # buffers. That is a real effect on its own and needs its own
                # control -- run the same --selftrain-at with --selftrain-frac 0
                # to measure the optimizer reset alone before attributing
                # anything to the synthetic data.
                _train_block(local_train, split_epoch, 0)
                local_train = _generate_and_mix(
                    model, cid, r, at_epoch=split_epoch,
                    source=selftrain_source,
                    global_state=global_state) or local_train
                _train_block(local_train, local_epochs - split_epoch,
                             split_epoch)
            elif fedrep_alternate:
                # Head first, against the body the server just sent, so the
                # head the client keeps is fitted to a GLOBAL body. Then the
                # body, with that head frozen. Same passes over the data as the
                # joint path: local_epochs - 1 for the head, 1 for the body.
                # train_model optimises only requires_grad parameters, so the
                # freeze is what selects each block.
                head_ep = local_epochs - 1
                try:
                    for n_, p_ in model.named_parameters():
                        p_.requires_grad_(n_ in pkeys)
                    _train_block(local_train, head_ep, 0)
                    for n_, p_ in model.named_parameters():
                        p_.requires_grad_(n_ not in pkeys)
                    _train_block(local_train, 1, head_ep)
                finally:
                    for p_ in model.parameters():
                        p_.requires_grad_(True)
            else:
                # Unsplit: one call, exactly as before. at=0 must stay on this
                # path so existing runs remain reproducible bit for bit.
                _train_block(local_train, local_epochs, 0)
            snap = _snapshot(model)
            shared, mine = split_parts(snap, pkeys)
            if pkeys:
                personal[cid] = mine
            if comm is not None:
                shared = {k: shared[k] for k in comm}
            # Only the shared half reaches the average; average_state_dicts is
            # untouched and averages whatever it is handed.
            snapshots.append(shared)
            weights.append(1.0 if weighting == "uniform" else float(sizes[cid]))
            lr_ = getattr(model, "last_rate", None)
            log(f"  round {r + 1}/{n_rounds}  client {cid}  (n={sizes[cid]}) done"
                + (f"  emitted/observed {lr_['pred_over_obs']:.3f}" if lr_ else ""))

        if writer is not None and round_final_losses:
            writer.add_scalar("loss_train/round_mean",
                              sum(round_final_losses) / len(round_final_losses), r)

        if prox_mu > 0.0:
            clear_prox_ref(model.halo_model)
        if dp_clip > 0:
            # DP-FedAvg: clip each client's DELTA against the weights they were
            # broadcast this round, average, then noise. `round_ref` is that
            # broadcast state -- not the post-aggregation state -- or the delta
            # is measured against the wrong origin.
            if dp_mech == "sgm":
                avg_shared, dp_norms = sgm_aggregate(
                    snapshots, round_ref, clip=dp_clip, sigma=dp_sigma,
                    sketch_b=dp_sketch_b, keys=dp_keys, round_seed=r,
                    weighting=weighting, device=device)
            else:
                avg_shared, dp_norms = dp_aggregate(
                    snapshots, round_ref, clip=dp_clip, sigma=dp_sigma,
                    weighting=weighting, keys=dp_keys)
            log(f"  [dp] round {r + 1} client update norms: "
                + " ".join(f"{n:.3f}" for n in dp_norms)
                + f"   clip={dp_clip:g} sigma={dp_sigma:g}"
                + f"  clipped={sum(1 for n in dp_norms if n > dp_clip)}/{len(dp_norms)}")
            if dp_adaptive > 0:
                # Andrew et al. (2021): move C toward a target quantile of the
                # observed norms. dp_clip is rebound here, so every later round
                # uses the updated value -- that is the whole mechanism.
                dp_clip, b_bar = adapt_clip(
                    dp_clip, dp_norms, target=dp_adaptive,
                    lr=dp_adapt_lr, sigma_b=dp_sigma_b)
                log(f"  [dp] adaptive: {b_bar:.3f} of clients under the old "
                    f"clip (target {dp_adaptive:g}) -> new clip {dp_clip:.4f}")
        else:
            avg_shared = average_state_dicts(snapshots, weights)
        if pkeys:
            # `global_state` must stay a COMPLETE, loadable state dict: the
            # model is loaded from it below, and the checkpoint has to match the
            # architecture. Fill the personal keys with their average purely so
            # the shape is right -- the loop never reads them back, each client
            # uses its own slice from `personal`. Said out loud because a reader
            # of fedavg_state.pt would otherwise take those keys at face value.
            personal_avg = average_state_dicts(
                [personal[cid] for cid in client_ids], weights)
            global_state = merge_parts(avg_shared, personal_avg)
        elif comm is not None:
            global_state = {**global_state, **avg_shared}
            moved = math.sqrt(sum(
                (global_state[k] - comm_start[k]).pow(2).sum().item()
                for k in comm))
            log(f"  round {r + 1}: adapter moved {moved:.4f} (L2 from start)")
            if writer is not None:
                writer.add_scalar("adapter_moved", moved, r + 1)
        else:
            global_state = avg_shared
        model.load_state_dict(global_state)
        is_last = (r + 1) == n_rounds
        if ckpt_path and ((r + 1) % ckpt_every == 0 or is_last):
            _save_ckpt(ckpt_path, r + 1, global_state, fingerprint,
                       extra={"personal_states": personal} if pkeys else None)
            log(f"round {r + 1}/{n_rounds} aggregated + checkpointed -> {ckpt_path}")
        else:
            log(f"round {r + 1}/{n_rounds} aggregated")
        log_gpu(writer, r + 1)

        # Held-out loss is a DIAGNOSTIC, not just an early-stopping input.
        # Gating it on `stopper` meant every --no-early-stop run -- which is
        # every arm in this project -- recorded no validation curve at all, so
        # a run that diverged mid-training looked identical to one that did
        # not until the end-of-run prevalence metrics landed. Self-training in
        # particular can only be watched here: its failure is the model drifting
        # onto its own output, which shows up as val loss rising while train
        # loss keeps falling.
        if val_dataset is not None:
            vl = val_loss(model, val_dataset, device)
            if writer is not None:
                writer.add_scalar("loss_val/global", vl, r + 1)
            if stopper is None:
                log(f"  round {r + 1}: val {vl:.4f}")
            else:
                stop = stopper.step(vl, state=global_state)
                log(f"  round {r + 1}: val {vl:.4f}  "
                    f"(best {stopper.best_score:.4f} at round "
                    f"{stopper.best_step})")
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

    # The per-client slices ride out on the model rather than changing the
    # return type, which every existing caller depends on. Empty unless a
    # key-partitioned regime asked for them.
    model._personal_states = personal
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
                      irm_warmup: int = 0,
                      irm_schedule: str = "step",
                      xm_k: int = 1) -> object:
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

        if val_dataset is None:
            log(f"  centralized  epoch {done}/{total_epochs}  "
                f"loss={mean_loss:.4f}")
            return None
        vl = val_loss(model, val_dataset, device)
        if writer is not None:
            writer.add_scalar("loss_val/pooled", vl, done - 1)
            log_irm(writer, model, done - 1, "pooled")
            log_xm(writer, model, done - 1, "pooled")
        # Same reasoning as the fedavg path: log the curve even with no stopper.
        if stopper is None:
            log(f"  centralized  epoch {done}/{total_epochs}  "
                f"loss={mean_loss:.4f}  val={vl:.4f}")
            return None
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
                                                   irm_warmup, irm_schedule,
                                                   epochs),
                      xm_k=xm_k)
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
    irm_schedule: str = "step",
    xm_k: int = 1,
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
                log_xm(writer, m, done - 1, f"hospital_{hid}")
                log_gpu(writer, done - 1, tag="gpu_local")
            if path and (done % ckpt_every == 0 or done == epochs):
                _save_ckpt(path, done, _snapshot(m), fingerprint,
                           key="completed_epochs")
            score = val_loss(m, val_ds, device) if use_val else mean_loss
            if writer is not None and use_val:
                writer.add_scalar(f"loss_val/hospital_{hid}", score, done - 1)
            if stopper is None:
                if use_val:
                    log(f"  local {hid}  epoch {done}/{epochs}  val={score:.4f}")
                return None
            stop = stopper.step(score, state=lambda: _snapshot(m))
            return False if stop else None

        m._epochs = epochs - start_epoch
        m.train_model(train_subset, val_dataset=None, device=device,
                      on_epoch_end=_on_epoch_end,
                      sample_weight_fn=sample_weight_fn,
                      irm_rho=lambda e, s0=start_epoch: irm_rho_at(
                          s0 + e, irm_rho, irm_warmup, irm_schedule, epochs),
                      xm_k=xm_k)
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


def _adapter_state(m) -> Dict[str, torch.Tensor]:
    """The trainable (adapter) parameters of ``m``, keyed as in ``halo_model``."""
    return {n: p.detach().cpu().clone()
            for n, p in m.halo_model.named_parameters() if p.requires_grad}


def _load_adapter(m, state: Dict[str, torch.Tensor], keys) -> None:
    """Load adapter weights only, and prove every one of them landed.

    ``strict=False`` is required -- the state holds the adapter, not the trunk
    -- and it is exactly the setting that lets a renamed key vanish silently.
    So both directions are checked: nothing unexpected, nothing missing.
    """
    _, unexpected = m.halo_model.load_state_dict(state, strict=False)
    if unexpected:
        raise KeyError(f"adapter state has keys the model lacks: "
                       f"{unexpected[:3]}")
    absent = set(keys) - set(state)
    if absent:
        raise KeyError(f"adapter state is missing {len(absent)} key(s), e.g. "
                       f"{sorted(absent)[:3]}")


def run_perada(
    global_state: Dict[str, torch.Tensor],
    build_model: Callable[[], object],
    clients: Dict[str, object],
    rounds: int,
    processor,
    lam: float = 0.1,
    kd_steps: int = 10,
    kd_pool: int = 1000,
    kd_batch: int = 64,
    adapter_rank: int = 8,
    weighting: str = "sample",
    device: str = "cpu",
    log: Callable[[str], None] = print,
    writer=None,
    ckpt_dir: str = None,
    fingerprint: dict = None,
) -> Dict[str, object]:
    """PerAda (Xie et al., CVPR 2024) over a frozen federated trunk.

    The paper's algorithm, round by round, with only the adapters moving:

      * client m, from the broadcast global adapter w:
          local adapter   theta_m <- w, one epoch of BCE           -> sent up
          personal adapter v_m, one epoch of BCE
                          + (lambda/2) ||v_m - w||^2               -> kept
      * server: w <- average of the theta_m, then ``kd_steps`` steps of
        distillation -- student the global adapter, teacher the uniform
        ensemble of the clients' local adapters, loss the per-code binary KL.

    Each hospital's released generator is trunk + v_m.

    THREE DEPARTURES FROM THE PAPER, all forced by the setting:

      1. No pretrained backbone exists for EHR generation, so the frozen trunk
         is ``global_state`` -- the output of the FedAvg rounds that precede
         this call. It is never updated here.
      2. The paper distills on a public auxiliary set (CIFAR-100 for
         CIFAR-10). There is no public EHR set, so the server distills on
         ``kd_pool`` records it samples ONCE from the trunk before round 1 and
         reuses. Sampling is the expensive step, autoregressive and
         sequential; the fixed pool keeps it a one-off, and a fixed set is
         also what the paper's auxiliary data is.
      3. ``kd_steps`` defaults to 10 against the paper's 100. As it goes to 0
         this is FedAvg over adapters plus a Ditto-style personal adapter; read
         a result near Ditto's with that in mind.

    Budget: one epoch for theta_m plus one for v_m, so a round costs a client
    two passes over its data -- the same as a FedAvg round at local_epochs=2,
    which is what lets the caller split one ``total_epochs`` across trunk
    rounds and PerAda rounds.

    Not resumable mid-way. A rerun resumes the trunk from fedavg_state.pt and
    repeats every PerAda round from round 1; it never continues from a partial
    adapter state it cannot prove is consistent.

    Returns:
        ``{hospital_id: model}``, each trunk + its personal adapter. With
        ``ckpt_dir`` set, also writes ``ft_<hid>.pt`` per hospital, in the same
        form finetune_local does, so generate.py loads them unchanged.
    """
    from pyhealth.datasets import get_dataloader
    from pyhealth.models.generators.adapters import apply_adapter
    from pyhealth.models.generators.fedreg import binary_kl, get_prox_ref

    def fresh(trainable: bool = True):
        m = build_model()
        m.load_state_dict(global_state)            # the frozen trunk u
        apply_adapter(m.halo_model, "lora_head", adapter_rank)
        if not trainable:
            for p_ in m.halo_model.parameters():
                p_.requires_grad_(False)
            m.halo_model.eval()
        m.to(device)
        return m

    # The server's distillation data: sampled once from the trunk, BEFORE any
    # adapter exists, so it depends on nothing the rounds change.
    gen = build_model()
    gen.load_state_dict(global_state)
    raw = gen.generate(num_samples=kd_pool, device=device)
    kept = {f"kd_{i}": p["visits"] for i, p in enumerate(raw)
            if p.get("visits")}
    del gen, raw
    if not kept:
        raise RuntimeError("the trunk generated no non-empty records for the "
                           "distillation pool")
    kd_ds = _to_sample_dataset({"server": kept}, processor, "perada_kd_pool")
    log(f"PerAda: {len(kept)} server-side distillation records sampled once "
        f"from the frozen trunk; {kd_steps} KD steps/round at batch {kd_batch}")

    worker = fresh()                   # reused for every client, every round:
    student = fresh()                  # the trunk is frozen, so only the
    teacher = fresh(trainable=False)   # adapter ever differs between uses
    w = _adapter_state(worker)         # w^0 is the LoRA initialisation
    keys = set(w)
    n_ad = sum(v.numel() for v in w.values())
    n_all = sum(p.numel() for p in worker.halo_model.parameters())
    log(f"PerAda: {rounds} rounds over a frozen trunk; adapter lora_head "
        f"r={adapter_rank}, {n_ad:,} of {n_all:,} params "
        f"({100 * n_ad / n_all:.2f}%), lambda={lam:g}")
    personal = {hid: {k: v.clone() for k, v in w.items()} for hid in clients}

    kd_loader = get_dataloader(kd_ds, batch_size=kd_batch, shuffle=True)
    kd_iter = iter(kd_loader)

    for t in range(1, rounds + 1):
        thetas, weights, drift = [], [], []
        for hid, ds in clients.items():
            _load_adapter(worker, w, keys)
            worker.train_model(ds, val_dataset=None, device=device)
            thetas.append(_adapter_state(worker))
            weights.append(float(len(ds)) if weighting == "sample" else 1.0)

            _load_adapter(worker, personal[hid], keys)
            set_prox_ref(worker.halo_model, w)
            worker.train_model(ds, val_dataset=None, device=device,
                               prox_mu=lam)
            drift.append(get_prox_ref(worker.halo_model).distance())
            clear_prox_ref(worker.halo_model)
            personal[hid] = _adapter_state(worker)

        w = average_state_dicts(thetas, weights)

        kd_first = kd_last = float("nan")
        if kd_steps > 0:
            _load_adapter(student, w, keys)
            student.halo_model.train()
            opt = torch.optim.Adam(
                [p_ for p_ in student.halo_model.parameters()
                 if p_.requires_grad], lr=student._lr)
            for step in range(kd_steps):
                try:
                    batch = next(kd_iter)
                except StopIteration:
                    kd_iter = iter(kd_loader)
                    batch = next(kd_iter)
                ehr, mask = student._encode_visits(batch["visits"].to(device))
                with torch.no_grad():
                    target = None
                    for th in thetas:
                        _load_adapter(teacher, th, keys)
                        _, p_t, _ = teacher.halo_model(
                            ehr, position_ids=None, ehr_labels=ehr,
                            ehr_masks=mask)
                        target = p_t if target is None else target + p_t
                    target = target / len(thetas)
                _, p_s, _ = student.halo_model(
                    ehr, position_ids=None, ehr_labels=ehr, ehr_masks=mask)
                loss = binary_kl(p_s, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step == 0:
                    kd_first = loss.item()
                kd_last = loss.item()
            w = _adapter_state(student)

        mean_drift = float(np.mean(drift))
        log(f"PerAda round {t}/{rounds}: mean ||v_m - w|| = {mean_drift:.4f}, "
            f"KD {kd_first:.5f} -> {kd_last:.5f}")
        if writer is not None:
            writer.add_scalar("perada/personal_drift", mean_drift, t)
            if kd_steps > 0:
                writer.add_scalar("perada/kd_loss", kd_last, t)

    models: Dict[str, object] = {}
    for hid in clients:
        m = fresh()
        _load_adapter(m, personal[hid], keys)
        models[hid] = m
        if ckpt_dir:
            path = os.path.join(ckpt_dir, f"ft_{hid}.pt")
            _save_ckpt(path, 1, _snapshot(m), fingerprint,
                       key="completed_finetune")
            log(f"  PerAda hospital {hid} -> {path}")
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
    irm_schedule: str = "step",
    xm_k: int = 1,
    adapter: str = "none",
    adapter_rank: int = 8,
    adapter_mu: float = 0.0,
    adapter_l1: float = 0.0,
    adapter_sparsity: float = 0.0,
    adapter_iht_every: int = 1,
    adapter_optim: str = "adam",
    adapter_lr: float = 0.0,
    ditto_lambda: float = 0.0,
    kd_lambda: float = 0.0,
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
    # Fail here rather than train eight hospitals and report a sparsity that
    # was never enforced: both knobs need the delta reference that only the
    # sparse variants install, and silently ignoring them is the expensive
    # failure mode.
    if (adapter_l1 > 0.0 or adapter_sparsity > 0.0) and \
            adapter not in ("last_mlp_l1", "last_mlp_iht"):
        raise ValueError(
            f"--adapter-l1/--adapter-sparsity need a delta-sparse variant, but "
            f"--adapter is {adapter!r}. Use last_mlp_l1 (proximal L1) or "
            f"last_mlp_iht (top-k projection).")
    if adapter == "last_mlp_l1" and adapter_l1 <= 0.0:
        raise ValueError("--adapter last_mlp_l1 needs --adapter-l1 > 0, "
                         "otherwise it is just last_mlp under another name.")
    if adapter == "last_mlp_iht" and adapter_sparsity <= 0.0:
        raise ValueError("--adapter last_mlp_iht needs --adapter-sparsity > 0, "
                         "otherwise it is just last_mlp under another name.")

    if kd_lambda > 0.0 and (xm_k > 1 or irm_rho > 0.0):
        raise ValueError(
            "--kd-lambda cannot be combined with --xm-k > 1 or --irm-rho > 0: "
            "three objectives over the same batch with no defined composition. "
            "Run them as separate arms.")

    # PerAda's teacher is the frozen final global -- identical for all eight
    # hospitals, so build it once rather than per site.
    teacher = None
    if kd_lambda > 0.0:
        t = build_model()
        t.load_state_dict(global_state)
        t.to(device)
        t.halo_model.eval()          # no dropout noise in the distillation target
        for p_ in t.halo_model.parameters():
            p_.requires_grad_(False)
        teacher = t.halo_model
        log(f"PerAda: KD toward the frozen global, lambda={kd_lambda:g}")

    models: Dict[str, object] = {}
    for hid, train_subset in clients.items():
        log(f"FedAvg+FT: fine-tuning hospital {hid} from global "
            f"(n={len(train_subset)})")
        m = build_model()
        m.load_state_dict(global_state)  # warm start from the federated global
        # Adapter AFTER load_state_dict: apply_adapter wraps modules, which
        # changes parameter names, and a wrapped model will not accept the
        # trunk's plain state dict. Order matters and getting it wrong raises
        # a key mismatch rather than failing quietly.
        if adapter != "none":
            from pyhealth.models.generators.adapters import apply_adapter
            trainable = apply_adapter(m.halo_model, adapter, adapter_rank)
            n_tr = sum(p.numel() for p in trainable)
            n_all = sum(p.numel() for p in m.halo_model.parameters())
            extra = ""
            if adapter_mu:
                extra += f", mu={adapter_mu:g}"
            if adapter_l1:
                extra += f", l1={adapter_l1:g}"
            if adapter_sparsity:
                extra += (f", density={adapter_sparsity:g} every "
                          f"{adapter_iht_every}ep")
            if adapter_optim != "adam":
                extra += f", optim={adapter_optim}"
            if adapter_lr:
                extra += f", lr={adapter_lr:g}"
            log(f"  {hid}: adapter {adapter} r={adapter_rank} -- training "
                f"{n_tr:,} of {n_all:,} params ({100 * n_tr / n_all:.2f}%)"
                + extra)

        if ditto_lambda > 0.0:
            # Ditto: the reference is the FROZEN final global w*, attached after
            # the adapter so trainable_only sees the final parameter set.
            set_prox_ref(m.halo_model, global_state)

        val_ds = (val_clients or {}).get(hid)
        stopper, use_val = make_site_stopper(
            hid, len(val_ds) if val_ds is not None else 0, es_cfg, log)

        def _on_epoch_end(epoch, mean_loss, hid=hid, m=m, val_ds=val_ds,
                          stopper=stopper, use_val=use_val):
            if writer is not None:
                writer.add_scalar(f"loss_ft/hospital_{hid}", mean_loss, epoch)
                log_irm(writer, m, epoch, f"ft_{hid}")
                log_sparse(writer, m, epoch, f"ft_{hid}")
                log_xm(writer, m, epoch, f"ft_{hid}")
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
                      # The IRM penalty NEVER applies during fine-tuning, for
                      # any variant. IRM enforces invariance across sites and
                      # local fine-tuning deliberately breaks it, so running both
                      # on the same stage has them pull against each other --
                      # true whether the update is low-rank or not. --irm-rho on
                      # a fedavg_ft run therefore means "IRM trunk, plain local
                      # fine-tuning", which is the composition worth testing.
                      irm_rho=0.0,
                      # Ditto: pull the personal model back toward the frozen
                      # final global. lambda=0 leaves plain fine-tuning.
                      prox_mu=ditto_lambda,
                      kd_lambda=kd_lambda,
                      kd_teacher=teacher,
                      adapter_mu=adapter_mu,
                      adapter_l1=adapter_l1,
                      adapter_sparsity=adapter_sparsity,
                      adapter_iht_every=adapter_iht_every,
                      adapter_optim=adapter_optim,
                      adapter_lr=adapter_lr)
        if adapter in ("last_mlp_l1", "last_mlp_iht"):
            # To stdout, not only TensorBoard: this is the run's evidence that
            # the constraint was actually enforced, and it must survive a
            # --no-tb smoke run. A density of 1.0 here means the proximal step
            # or the projection never fired and the arm is plain last_mlp.
            from pyhealth.models.generators.adapters import delta_stats
            st = delta_stats(m.halo_model)
            log(f"  {hid}: delta density {st['nnz_frac']:.4f} "
                f"({st['nnz']:,}/{st['n']:,} coords), "
                f"||D||_1={st['l1']:.4g}, ||D||_inf={st['linf']:.4g}")

        if stopper is not None:
            log(f"  {hid}: {stopper.summary()}")
            # The warm start means epoch 0 can already be the best the site
            # gets: fine-tuning a converged global on 142 records often makes
            # val loss worse immediately, and keeping the best is what stops
            # that from silently becoming the reported fedavg_ft result.
            if stopper.best_state is not None:
                m.load_state_dict(stopper.best_state)
                # The restored epoch may not have been a projection epoch, in
                # which case the "sparse" model just written to disk is dense.
                # Re-project so the checkpoint matches what the run claims.
                if adapter_sparsity > 0.0:
                    from pyhealth.models.generators.adapters import (
                        delta_stats, hard_threshold_)
                    hard_threshold_(m.halo_model, adapter_sparsity)
                    log(f"  {hid}: re-projected restored best state -- "
                        f"density {delta_stats(m.halo_model)['nnz_frac']:.4f}")
        if ditto_lambda > 0.0:
            from pyhealth.models.generators.fedreg import get_prox_ref
            # How far lambda actually let this site move. Log it: the method IS
            # a claim about this distance, and a value that does not shrink as
            # lambda rises means the penalty is not biting.
            log(f"  {hid}: ||v - w*|| = "
                f"{get_prox_ref(m.halo_model).distance():.4f} "
                f"(lambda={ditto_lambda:g})")
            clear_prox_ref(m.halo_model)
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
    if cfg.get("print_save_dir"):
        # Nothing but the name, on stdout, so `dir=$(... --print-save-dir)` in a
        # shell script gets a clean value.
        print(f"_outputs/{cfg['run_name']}_save")
        raise SystemExit(0)
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
    # Self-training needs the RAW per-hospital trajectories, not just the
    # SampleDatasets: the mixture is rebuilt each round from real + freshly
    # generated patients, through the same pinned processor.
    client_trajectories = (read_trajectories(cache, "train")
                           if cfg.get("selftrain_frac", 0.0) > 0 else None)
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
    if regime == "perada_fl":
        bad = []
        if cfg["adapter"] != "lora_head":
            bad.append(f"--adapter lora_head (got {cfg['adapter']!r}); "
                       "generate.py re-applies the adapter from the config")
        if cfg["ft_epochs"] % 2:
            bad.append(f"an even --ft-epochs (got {cfg['ft_epochs']}): each "
                       "PerAda round spends 2 passes, so rounds = ft_epochs/2")
        if cfg["latent_dim"] or cfg["xm_k"] > 1 or cfg["irm_rho"] > 0:
            bad.append("no --latent-dim / --xm-k / --irm-rho: the baseline "
                       "is PerAda alone")
        if cfg["kd_lambda"] or cfg["ditto_lambda"]:
            bad.append("no --kd-lambda / --ditto-lambda: those belong to "
                       "the post-hoc regimes; use --perada-lambda")
        if bad:
            raise ValueError("--regime perada_fl needs " + "; ".join(bad))
    if cfg["dp_mech"] == "sgm" and regime != "fedavg":
        raise ValueError("--dp-mech sgm is only wired into --regime fedavg")
    if regime in FT_REGIMES:
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
            print(f"[budget] {regime} passes/hospital = {spent} "
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
            dropout=cfg["dropout"],
            latent_dim=cfg["latent_dim"],
            save_dir=save_dir,
        )

    # Warm start (--init-from) and a whole-run frozen trunk (--adapter outside
    # the FT regimes). Both happen in prepare(), which every non-FT arm's model
    # passes through, so centralized, local and fedavg start from the same
    # weights and freeze the same parameters.
    init_state = None
    if cfg["init_from"]:
        init_state = load_init_state(
            cfg["init_from"], cfg,
            sample_dataset.input_processors["visits"].code_vocab)
        # HALO.train_model reloads save_dir/halo_model at the top of EVERY call
        # if it exists, which would silently replace the warm start.
        stale = os.path.join(save_dir, "halo_model")
        if os.path.exists(stale):
            raise ValueError(
                f"{stale} exists and HALO.train_model would load it over the "
                f"--init-from weights. Delete it or use a fresh --run-name.")
    freeze_trunk = (cfg["adapter"] != "none"
                    and regime in ("centralized", "local", "fedavg"))
    if freeze_trunk:
        bad = []
        if init_state is None:
            bad.append("--init-from: a frozen trunk that was never trained is "
                       "random features, not a pretrained model")
        if cfg["adapter"] not in ("lora_attn", "last_mlp", "lora_head"):
            bad.append(f"a dense adapter (got {cfg['adapter']!r}); the sparse "
                       "variants' constraint lives in finetune_local's loop")
        if (cfg["adapter_mu"] or cfg["adapter_l1"] or cfg["adapter_sparsity"]
                or cfg["adapter_optim"] != "adam"):
            bad.append("no --adapter-mu/-l1/-sparsity/-optim: those are "
                       "fine-tuning-stage knobs this path does not apply")
        if cfg["adapter_lr"] and regime != "fedavg":
            bad.append("no --adapter-lr: only the fedavg path applies it here")
        if cfg["dp_clip"] and cfg["dp_mech"] != "sgm":
            bad.append("--dp-mech sgm for --dp-clip: Gaussian DP over the "
                       "adapter slice is not implemented")
        if bad:
            raise ValueError(f"--adapter with --regime {regime} needs "
                             + "; ".join(bad))

    def prepare(m):
        """Load the warm start, then freeze the trunk if this run adapts it."""
        if init_state is not None:
            m.load_state_dict(init_state)
        # After the load: apply_adapter wraps modules and renames their keys,
        # so the trunk's plain state dict would no longer fit.
        if freeze_trunk:
            from pyhealth.models.generators.adapters import apply_adapter
            apply_adapter(m.halo_model, cfg["adapter"], cfg["adapter_rank"])
        return m

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
            models = train_local(lambda: prepare(build_model(total_epochs)),
                                 clients,
                                 device=device, writer=writer,
                                 total_epochs=total_epochs, ckpt_dir=save_dir,
                                 ckpt_every=cfg["ckpt_every"],
                                 resume=cfg["resume"], fingerprint=fingerprint,
                                 val_clients=client_vals, es_cfg=es_cfg,
                                 sample_weight_fn=sample_weight_fn,
                                 irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"],
                    irm_schedule=cfg["irm_schedule"], xm_k=cfg["xm_k"])
            first = next(iter(models.values()))
            num_params = sum(p.numel() for p in first.parameters())
            n_train = sum(p.numel() for p in first.parameters()
                          if p.requires_grad)
            print(f"Each model: {num_params} parameters, {n_train} trainable")
            # STEP 5: every local model generates the SAME number of patients;
            # the pooled set is then a proportional subsample of those.
            synthetic, client_synth = generate_local(
                models, sizes, cfg["num_synth"], cfg["synth_per_hospital"],
                device=device)
        else:
            # centralized trains the single model for the full budget; fedavg and
            # fedavg_ft use local_epochs per round (the FedAvg local step).
            epochs = total_epochs if regime == "centralized" else cfg["local_epochs"]
            model = prepare(build_model(epochs))
            num_params = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
            print(f"\nModel initialized with {num_params} parameters, "
                  f"{n_train} trainable")
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
                    val_dataset=pooled_val,
                    stopper=(EarlyStopper(cfg["es_patience"],
                                          cfg["es_min_delta"],
                                          cfg["es_min_steps"])
                             if es_cfg else None),
                    sample_weight_fn=sample_weight_fn,
                    irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"],
                    irm_schedule=cfg["irm_schedule"], xm_k=cfg["xm_k"])
            else:  # every FedAvg-shaped regime starts with a FedAvg run
                pmode = PERSONAL_MODE.get(regime, "none")
                print(f"Regime: {regime} -- {len(clients)} clients, "
                      f"{cfg['local_epochs']} local epochs x {fed_rounds} rounds"
                      + (f", then {cfg['ft_epochs']} fine-tuning epochs/hospital "
                         f"(total {total_epochs} passes/hospital)"
                         if regime in FT_REGIMES else "")
                      + (f", personal={pmode}" if pmode != "none" else "")
                      + (f", prox mu={cfg['fedprox_mu']:g}"
                         if cfg["fedprox_mu"] else ""))
                # Checkpoint aggregated weights every --ckpt-every rounds; --resume
                # continues from the last completed round (fedavg only).
                ckpt_path = os.path.join(save_dir, "fedavg_state.pt")
                run_fedavg(model, clients, n_rounds=fed_rounds, device=device,
                           selftrain_frac=cfg["selftrain_frac"],
                           selftrain_start=cfg["selftrain_start"],
                           selftrain_at=cfg["selftrain_at"],
                           selftrain_source=cfg["selftrain_source"],
                           client_trajectories=client_trajectories,
                           processor=sample_dataset.input_processors["visits"],
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
                           val_dataset=pooled_val,
                           stopper=(EarlyStopper(cfg["es_patience"],
                                                 cfg["es_min_delta"],
                                                 cfg["es_min_steps"])
                                    if es_cfg else None),
                           sample_weight_fn=sample_weight_fn,
                           irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"],
                           irm_schedule=cfg["irm_schedule"], xm_k=cfg["xm_k"],
                           personal_mode=pmode,
                           prox_mu=cfg["fedprox_mu"],
                           dp_clip=cfg.get("dp_clip") or 0.0,
                           dp_sigma=cfg.get("dp_sigma") or 0.0,
                           dp_adaptive=cfg.get("dp_adaptive") or 0.0,
                           dp_sigma_b=cfg.get("dp_sigma_b") or 0.0,
                           dp_adapt_lr=cfg.get("dp_adapt_lr") or 1.0,
                           rate_mu=cfg.get("rate_mu") or 0.0,
                           fedrep_alternate=bool(cfg.get("fedrep_alternate")),
                           # Only a frozen-trunk run: under fedavg_ft this knob
                           # belongs to the fine-tuning stage, not the rounds.
                           adapter_lr=cfg["adapter_lr"] if freeze_trunk else 0.0,
                           dp_mech=cfg["dp_mech"],
                           dp_sketch_b=cfg["dp_sketch_b"])
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
            if regime in SHARED_GENERATOR_REGIMES:
                want = max(want, len(clients) * cfg["synth_per_hospital"])
            full = model.generate(num_samples=want, device=device)
            client_synth = {hid: full for hid in clients}
            # The POOLED set stays num_synth for every regime, so Test 1's
            # prevalence resolution (1/num_synth) is identical across arms and
            # its numbers stay comparable. Only the per-hospital sets scale.
            synthetic = full[:cfg["num_synth"]]
            if regime in PERSONAL_MODE:
                # FedRep / LG-FedAvg: there is no fine-tuning phase. Each
                # hospital's model IS the shared half plus the half it kept out
                # of the average, which run_fedavg has been carrying all along.
                personal = getattr(model, "_personal_states", {})
                if not personal:
                    raise RuntimeError(
                        f"{regime} finished with no per-client weights; the "
                        "partitioned path did not run.")
                global_state = _snapshot(model)
                part_models = {}
                for hid in clients:
                    m = build_model(cfg["local_epochs"])
                    m.load_state_dict(merge_parts(global_state, personal[hid]))
                    part_models[hid] = m
                    _save_ckpt(os.path.join(save_dir, f"ft_{hid}.pt"), 1,
                               m.state_dict(), fingerprint,
                               key="completed_finetune")
                synthetic, client_synth = generate_local(
                    part_models, sizes, cfg["num_synth"],
                    cfg["synth_per_hospital"], device=device)
            elif regime == "perada_fl":
                ft_models = run_perada(
                    _snapshot(model), lambda: build_model(1), clients,
                    rounds=cfg["ft_epochs"] // 2,
                    processor=sample_dataset.input_processors["visits"],
                    lam=cfg["perada_lambda"],
                    kd_steps=cfg["perada_kd_steps"],
                    kd_pool=cfg["perada_kd_pool"],
                    kd_batch=cfg["perada_kd_batch"],
                    adapter_rank=cfg["adapter_rank"],
                    weighting=cfg["weighting"], device=device, writer=writer,
                    ckpt_dir=save_dir, fingerprint=fingerprint)
                _, client_synth = generate_local(
                    ft_models, sizes, cfg["num_synth"],
                    cfg["synth_per_hospital"], device=device)
            elif regime in FT_REGIMES:
                # Personalize: fine-tune the global model on each hospital, then
                # score each hospital against its OWN fine-tuned model in STEP 7.
                global_state = _snapshot(model)
                ft_models = finetune_local(
                    global_state, lambda: build_model(cfg["ft_epochs"]),
                    clients, device=device, writer=writer,
                    ckpt_dir=save_dir, fingerprint=fingerprint,
                    val_clients=client_vals, es_cfg=es_cfg,
                    sample_weight_fn=sample_weight_fn,
                    irm_rho=cfg["irm_rho"], irm_warmup=cfg["irm_warmup"],
                    irm_schedule=cfg["irm_schedule"], xm_k=cfg["xm_k"],
                    adapter=cfg["adapter"], adapter_rank=cfg["adapter_rank"],
                    adapter_mu=cfg["adapter_mu"],
                    adapter_l1=cfg["adapter_l1"],
                    adapter_sparsity=cfg["adapter_sparsity"],
                    adapter_iht_every=cfg["adapter_iht_every"],
                    adapter_optim=cfg["adapter_optim"],
                    adapter_lr=cfg["adapter_lr"],
                    ditto_lambda=cfg["ditto_lambda"],
                    kd_lambda=cfg["kd_lambda"])
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
        shared_generator=regime in SHARED_GENERATOR_REGIMES)
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
 