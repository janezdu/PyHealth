# Plan: feeding the generator its own output

**Status: round-level version IMPLEMENTED** as `--selftrain-frac` /
`--selftrain-start` (`train.py`, generation hook at ~line 1268). The
**intra-round** ("ouroboros") variant below is the proposal.

## The idea

Train HALO, sample synthetic patients from it, mix a *fraction* of those back into
its own training data, keep training, repeat. Never a full batch of synthetic —
every gradient step sees real records too.

## Why it is worth trying here specifically

Two reasons, and only the second is really about this project.

**The generic one:** more data. Site 429 holds 142 patients. Any augmentation
looks attractive at that size.

**The one with a mechanism:** in the *federated* variant, the synthetic data is
sampled from the **aggregated global**, so what a small hospital mixes into its
local step carries information from the other seven sites — without any raw
record leaving anyone's machine. That is federated distillation wearing a
generative-model hat, and it is a real answer to the problem this cohort has.
Plain self-training on a site's own model carries no new information at all; it
can only regularise.

It also points at the right stage. Test 2 found that **nothing** at the
fine-tuning stage moves downstream utility — full fine-tune, three adapters, and
the IRM-trunk versions all land where the un-fine-tuned global lands. The only
lever that showed any movement was the *training objective* (IRM). This proposal
is a training-stage intervention, which is where the evidence says to look.

## The risk that makes it interesting

Iteratively training a generator on its own output is known to collapse the
distribution — "model autophagy", the curse of recursion. Each generation
resamples from a slightly narrower estimate, and **the tails go first**.

This project is *about the tails*. Rare codes are the entire evaluation. So the
failure mode of self-training is precisely aligned with the thing we measure,
which cuts both ways: it is the reason to be careful, and it is why this cohort
is a good place to test it — we can actually see the collapse rather than infer
it.

The published stabiliser is exactly the constraint already asked for: keep real
data in the mixture at a fixed proportion rather than replacing it, and the
process stops being a pure feedback loop. Accumulating synthetic across
generations (rather than discarding the previous generation's) is reported to
help further. Both are cheap to vary and both should be arms.

## Design decisions

### 1. Mix at the BATCH level, not the dataset level

Dataset-level mixing at ratio alpha still produces all-synthetic batches by
chance. Batch-level mixing guarantees every gradient step is anchored on real
records, which is the literal reading of "never a full batch of generated data"
and the stronger version of the constraint. It needs a custom sampler, not a
concatenated dataset.

### 2. alpha — the synthetic fraction of each batch

`{0.0, 0.25, 0.5}`. 0.0 is the baseline and must be **compute-matched** (see
guards). Going above 0.5 abandons the constraint and is a different experiment.

### 3. Where in the pipeline

| variant | who generates | what it tests |
|---|---|---|
| **A · centralized** | the model itself | does self-augmentation help at all, federation aside |
| **B · federated local step** | the aggregated global | does cross-site knowledge transfer through synthetic data |
| C · fine-tuning only | the global | ruled out in advance by the Test 2 null; skip |

Run **A first as the control** and **B as the one with a mechanism**. A alone
cannot distinguish "augmentation helps" from "federation helps"; B minus A is
the federated contribution.

### 4. Refresh cadence

Regenerate from the current model every K rounds (start K = 5). Two knobs
interact: too frequent and the synthetic set is noise from an untrained model,
too rare and it is stale. Generate a *small* draw per refresh — this does not
need 8,000/hospital, and generation cost is what will dominate the run.

### 5. Replace vs accumulate

`replace` (only the newest generation's synthetic) and `accumulate` (keep all
previous). Accumulate is reported to resist collapse; it also grows memory
linearly. One arm each, at the best alpha from the A sweep.

## Guards — the things that fail silently

1. **Tail collapse must be watched per generation, not discovered at the end.**
   Log, every refresh: distinct codes appearing anywhere in the synthetic draw,
   mean codes per visit, and prevalence entropy. Collapse shows in the
   distinct-code count generations before it reaches Test 1. A monotone decline
   is the stop signal.

2. **Validation loss must be computed on REAL data only.** Training loss on a
   real+synthetic mixture is a different objective at every alpha and every
   generation, so it cannot be compared across arms or used for early stopping.
   This is the same trap the existing sample-weighting code already documents
   for weighted training.

3. **The baseline must be compute-matched.** Mixing synthetic in adds gradient
   steps. If the alpha = 0 arm trains for fewer steps, the comparison measures
   compute, not augmentation. Match total optimiser steps, not epochs.

4. **Synthetic must never be conditioned on val or test.** The generator that
   produces the mixture is trained on the train split only. Assert the
   provenance rather than assume it — this is the one error that would make
   every downstream number meaningless.

5. **Assert the batch composition.** "Every batch contains real records" is the
   whole premise; a sampler bug that silently produces all-synthetic batches
   would look like a training-dynamics result. Test it.

6. **Fix the bootstrap seed first.** `compute_prevalence_metrics` bootstraps
   unseeded (+/-0.019 Pearson, +/-0.079 R^2). This experiment's effects are
   likely smaller than that. Seeding it is a prerequisite, not a nicety.

## What counts as success

Improvement on **rare-code** metrics, **at the small sites**, **without tail
shrinkage across generations**. All three clauses matter:

- a gain on common codes only is not interesting for this cohort;
- a gain at site 458 (1,832 patients) does not address the problem;
- a gain that comes with a falling distinct-code count is borrowed against a
  collapse that has not arrived yet, and will not survive more generations.

Report the per-generation diagnostics alongside the metrics, always. A single
end-of-run number cannot distinguish "stable and better" from "two generations
away from collapse".

## Honest prior

Self-training a generator on its own samples is, in the literature,
neutral-at-best once real data is retained and clearly harmful when it is not.
Variant A will probably land inside the noise floor.

Variant B is the bet. It is not self-training in the circular sense — the
synthetic carries information the local site does not have — and it is the only
version with a reason to work. If A and B both land at zero, that is a clean
negative and the direction should be dropped rather than tuned.

## Rough cost

Generation, not training, will dominate: a refresh every K rounds means
~`n_rounds / K` generation passes per run. Centralized baselines on this cohort
run ~3h; expect a self-training arm to be meaningfully longer and size the
per-refresh draw down accordingly. Scope the first pass as variant A at three
alphas on `--profile full`, centralized only, before committing to the federated
variant.

## Prerequisites

- Seed the bootstrap in `pyhealth/metrics/generative/utility.py` (long deferred).
- A batch-level mixing sampler; `get_dataloader` currently has no hook for one.
- Per-generation diagnostics wired to TensorBoard, mirroring `log_sparse`.


---

# Variant: intra-round self-training (the ouroboros split)

**Status: proposal, 2026-09-08.** Nothing implemented.

## The idea

Split each client's local epochs at a boundary. With `local_epochs: 10`: run 8
epochs on real data only, then generate synthetic patients, then run the last 2
epochs on a real+synthetic mixture. One generation per round, as now -- but
placed *inside* local training rather than before it.

## How this differs from what exists

`--selftrain-frac` generates **once per round, before local training, from the
just-broadcast global**, and all `local_epochs` then run on the mixture. The
proposal moves the generation to an epoch boundary partway through local
training. That is a one-line change in placement and a fundamental change in
what the synthetic data *is*.

| | implemented | proposed |
|---|---|---|
| when | round start, before local epochs | after epoch `k` of `local_epochs` |
| generator | the aggregated global | the locally-adapted model |
| mixture applies to | all local epochs | the last `local_epochs - k` only |

## The tension this creates (read before implementing)

This note already argues that the *only* reason to expect anything from
self-training here is that the synthetic comes from the **global**, so a small
site's local step sees information from the other seven -- federated
distillation in a generative wrapper. Plain self-training on a site's own model
"carries no new information at all; it can only regularise."

Generating after 8 local epochs erodes exactly that. The model at epoch 8 is
global-initialised but locally adapted, so its samples are a blend whose
cross-site content **decays as `k` rises**. At `k = 0` this reduces to the
implemented behaviour; at `k = local_epochs` it is pure local self-training.

So `k` is not a tuning knob, it is the *independent variable of the whole
question*: how much local adaptation can the synthetic tolerate before it stops
carrying other sites' information. That is the experiment. It should be framed
that way rather than as "8 then 2".

The clean way to separate the two effects is a **generator-source arm**: at the
same `k`, generate from (a) the locally-adapted model and (b) the global
snapshot held aside from round start. Same schedule, same mixture, same cost --
the only difference is whose distribution the synthetic came from. If (a) and
(b) land together, the cross-site story is wrong and this is just
regularisation. If (b) wins, `k` should be small and the whole idea is an
argument for the implemented version.

## "Teacher forced" is already what happens

HALO's loop is autoregressive with `ehr_labels=batch_ehr` -- every step is
teacher-forced already, on real and synthetic alike. So "2 epochs of mixed
teacher forcing" means ordinary training on the mixture; no new mechanism.
Worth stating plainly because it removes an apparent design choice: the only
real choices are *when* to generate, *from what*, and *how much*.

Note what teacher forcing on your own generations means, though: it is the
textbook autophagy loop, with the model's own errors becoming targets. The
existing mitigations still apply and should stay -- fresh generation every
round, never accumulated, real data present in every batch.

## Relation to XM

Best-of-K (`--xm-k`) already does selection *at the loss level*: run K
stochastic passes, train on the lowest-loss one per patient. This proposal is
selection at the *data* level: sample from the model, train on the samples.
They compose, and the combination is worth one cell -- but they are not the
same mechanism and should not be conflated in the writeup.

If a latent is on, there is a sharper version available: generate the synthetic
pool by drawing K candidates per patient and keeping the best, so the pool is
made of the model's *confident* samples rather than typical ones. That is a
different (and more defensible) ouroboros than sampling the prior once. Not
proposed for the first pass -- noted so it is not reinvented later.

## Implementation sketch

1. `--selftrain-at-epoch k` (int, default 0 = current behaviour). Add to the
   run name (`_st0.5@8`) -- two runs differing only in `k` must not share a
   `save_dir`. Every knob that changes the trained model is in the name; this
   is no exception.
2. `--selftrain-source {global,local}` (default `global` = current behaviour).
   Requires keeping the round-start global snapshot alive through local
   training; `_snapshot(model)` already exists for aggregation.
3. Split the `model.train_model(...)` call at the client into two calls of
   `k` and `local_epochs - k` epochs, generating between them. The existing
   `on_epoch_end` callback and `irm_rho_at(...)` epoch arithmetic must keep
   counting through the split, or IRM's warmup schedule silently resets
   mid-round.
4. Reuse `mix_synthetic` and `_synth_health` unchanged. The
   `selftrain_distinct` / `selftrain_codes_per_visit` / `selftrain_kept`
   scalars are the collapse detector and must be logged per split, not per
   round -- a monotone fall in distinct codes is the stop signal, and it shows
   generations before any prevalence metric moves.

## Cost

Generation frequency is unchanged (once per client per round), so the
generation cost is the same as the implemented version. **The real cost is
`local_epochs`**: the board runs at `E2`, and this needs `E10` to have an 8/2
split at all -- 5x the local compute per round. At ~20-30 min per current run,
budget roughly 1.5-2.5 h per cell, not 30 min. Consider holding
`n_rounds * local_epochs` constant (e.g. `E10/R10` against the `E2/R50`
board) so total gradient steps match and the comparison is about placement
rather than budget.

## First grid (proposal)

Baseline is `control_plain_v1` plus the existing `st` runs. At fixed
`selftrain_frac: 0.5`, `local_epochs: 10`:

| axis | values | why |
|---|---|---|
| `selftrain_at_epoch` | 0, 8 | 0 reproduces the implemented behaviour exactly; 8 is the proposal |
| `selftrain_source` | global, local | the arm that separates distillation from regularisation |

4 cells. `(0, global)` should reproduce an existing `st` run -- if it does not,
the split refactor changed something and nothing else in the grid is readable.
That cell is the correctness check, not a result.

## Open questions

1. **Hold total gradient steps constant, or hold `n_rounds` constant?** Holding
   steps constant (`E10/R10`) makes this comparable to the board; holding
   rounds constant (`E10/R50`) is 5x the compute and confounds placement with
   budget. Default: hold steps constant.
2. **Does `k` want to be a fraction rather than an epoch count?** `k=8` means
   something different at `E10` than at `E2`. A fraction (`0.8`) survives a
   change in `local_epochs`; an int does not. Default: fraction, named in the
   run name as the resolved integer.
3. **Should the 2 mixed epochs use a lower LR?** Late-training epochs on
   partly-synthetic data at full LR can undo the real-data fit. Not proposed
   for the first pass; flagged because if the mixed epochs hurt, this is the
   first thing to try before concluding the idea fails.
