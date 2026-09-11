# Adapters: parameter-efficient per-hospital fine-tuning

`fedavg_ft` personalises the federated global by fine-tuning it at each site. The
default retrains all 6.17M HALO weights. At a hospital holding 142 patients that
is ~43,000 parameters per patient, and nothing stops the local run overwriting
the cross-site signal the federation just built.

An adapter freezes the trunk and trains a small delta instead. `--adapter`
selects which one; the trunk and the FedAvg rounds before fine-tuning are
untouched either way.

| variant | trains | params | % of full FT | prior on the update |
|---|---|---|---|---|
| `none` | everything | 6,230,872 | 100% | none — the baseline |
| `lora_attn` | LoRA on q,v of every attention block | 32,768 | 0.53% | low rank |
| `lora_head` | LoRA on the autoregressive code head | 37,760 | 0.61% | low rank |
| `last_mlp` | the final block's feedforward, in full | 525,568 | 8.49% | none (dense) |
| `last_mlp_l1` | same weights, L1-sparse delta | 525,568 | 8.49% × density | sparse |
| `last_mlp_iht` | same weights, top-k sparse delta | 525,568 | 8.49% × density | sparse |

`lora_attn` changes **how the model attends over visit history**; `lora_head`
changes **how codes are emitted given that history**. Test 1 scores emitted code
prevalence, so `lora_head` is aimed at the metric and `lora_attn` is the more
conservative edit. `last_mlp` is neither low-rank nor small — it is the "just
retrain the last layer" reference the others are trying to beat.

## Delta, not weights

The sparse variants sparsify **`D = W − W_global`**, not `W`. This is the whole
design, and getting it backwards is a different experiment:

* sparsify `W` — force the last MLP to be mostly zeros. A trained MLP is not
  sparse, so this deletes what the trunk knows. That is pruning.
* sparsify `D` — the site changes a few coordinates and **is the federated
  global everywhere else**.

Working on the delta also restores the property plain `last_mlp` lacks: "adapter
= 0 means be the federated global". That is what makes `--adapter-mu` a genuine
proximal term for these variants rather than weight decay toward the origin, and
it lines them up against LoRA as one question with two answers — LoRA
personalises along a few *directions*, these along a few *coordinates*.

`W_global` is stored as buffers on the model, so it round-trips through the
checkpoint and a finished run can still report its own sparsity. Plain
`last_mlp` deliberately does **not** get one: adding buffers would change its
state dict and the `last_mlp` checkpoints already on disk would stop loading.

## The three things that fail silently

1. **A subgradient L1 does not sparsify.** The subgradient of `|x|` is
   `sign(x)`, so the update chatters in a band of width `lr·λ` around zero and
   essentially never lands on it. The run regularises, reports 0% sparsity, and
   reads as a failed idea when it is a failed representation. `--adapter-l1`
   therefore applies a **proximal soft-threshold after the optimiser step**, not
   a term in the loss.
2. **Hard thresholding without masking the optimiser state does nothing.**
   Zeroing a weight while leaving Adam's `exp_avg` for that coordinate intact
   means the next step pushes it straight back off zero — the run ends dense
   while every projection looked like it worked. `hard_threshold_` masks
   `exp_avg`, `exp_avg_sq` and `momentum_buffer` alongside the weight.
3. **Sparsity is invisible in the loss.** A dense run and a 7%-dense one sit at
   indistinguishable training losses, so the loss curve cannot catch either bug
   above. `‖D‖₁` and the non-zero fraction are logged per epoch to TensorBoard
   (`sparse_l1/`, `sparse_nnz_frac/`) and printed per hospital to stdout, which
   is what makes the claim checkable.

`tests/core/test_halo_adapters.py` pins all three, including a guard test that
the un-masked optimiser really does re-inflate >90% of pruned coordinates — so
the masking assertion is testing something.

## Why an SGD arm exists

`--adapter-optim sgd` is a **correctness cross-check, not a better optimiser**:

* `lr·λ` is the *exact* proximal operator of the L1 under SGD (this is ISTA).
  Under Adam the correct per-coordinate threshold carries a `1/√v̂` factor, so
  prox-Adam is a heuristic wearing a proof's clothes.
* Plain SGD (momentum 0) carries no optimiser state, so failure 2 above cannot
  happen at all.

It needs `--adapter-lr`. The configured `1e-4` is an **Adam** learning rate; SGD
barely moves at it, and an untuned SGD arm underperforms for reasons that have
nothing to do with sparsity.

## The parameter-count ladder

`last_mlp` is 525,568 params, so density is what decides whether these compete
on cost or only on regularisation:

| density | effective params | vs `lora_head` (38K) |
|---|---|---|
| 0.5 | 263K | still 7× larger |
| 0.1 | 53K | same order |
| 0.07 | 37K | like-for-like |

Only at ~0.07 does "sparse vs low-rank at equal budget" become a sayable claim.
A win at 0.5 is a claim about *regularisation*, which is fine but different.

The L1 arm gets no target density — λ sets it indirectly, and where it lands is
a result rather than a setting.

## Measured: λ cannot control density in a federated setting

The most transferable thing the sparse work produced, and an argument for IHT
over L1 here rather than a tuning note.

The proximal step subtracts `lr·λ` **every step**, so the total shrinkage budget
is `N·lr·λ` — and N is the optimiser-step count, which scales with a site's
data. Achieved density at a single λ, on hilo8_random at `ft_epochs=20`:

| site | patients | λ=0.2 | λ=0.3 | λ=0.5 |
|---|---|---|---|---|
| 458 | 1,832 | 0.116 | 0.046 | **0.031** |
| 188 | 1,620 | 0.600 | 0.456 | 0.334 |
| 300 | 1,579 | 0.231 | 0.144 | 0.091 |
| 208 | 1,517 | 0.277 | 0.189 | 0.123 |
| 449 | 1,001 | 0.187 | 0.114 | 0.069 |
| 277 | 624 | 0.214 | 0.111 | 0.051 |
| 358 | 194 | 0.719 | 0.624 | 0.433 |
| 429 | 142 | 0.949 | 0.916 | **0.818** |

A **26× spread** at λ=0.5 — the smallest hospital stays 82% dense while the
largest reaches 3%. The middle is not monotone in size: site 188 sits at 0.334
despite being second largest, and it carries by far the largest `‖D‖₁` (712
against 458's 142). Density is set by the budget *relative to* the delta
magnitude, and both terms vary per site.

IHT hits its target on every site exactly — 0.0700 = 36,790/525,568 on all
eight. So **you cannot ask λ for a density, only for a shrinkage rate.** In a
cohort with 13× size heterogeneity that makes IHT the right instrument.

Calibration, if you do sweep λ: the transition is brutally sharp. At
`ft_epochs=20` on this cohort, λ=0.1 leaves the delta 99.3–99.9% dense and λ=1
drives it to **exactly zero** on every site, making those arms bit-identical to
the untuned federated global. The usable band is 0.2–0.5.

## What λ does do

Trace the specialist↔generalist tradeoff. More shrinkage pulls each site back
toward the federated global, monotonically in both directions:

| arm | specialist Pearson | generalist R² |
|---|---|---|
| λ=0.2 | 0.8630 | 0.5882 |
| λ=0.3 | 0.8509 | 0.6281 |
| λ=0.5 | 0.8459 | 0.6425 |
| *noise range* | *0.0105* | *0.0229* |

The generalist span is ~2.4× noise and monotone; everything else is inside
noise. Read it as mechanism-consistent rather than established — a random
ordering of three points is monotone one time in three.

At comparable density, sparse-by-shrinkage and sparse-by-projection land in the
same place: λ=0.5 scores 0.5012 / 0.6425 against `iht 0.07`'s 0.5098 / 0.6459.

## Everything here is near or under the noise floor

Scoring three **bit-identical** models (the degenerate λ=1/10/100 arms, which
had a delta of exactly zero) through independent generations gave the real
end-to-end spread: specialist R² **0.184**, specialist Pearson 0.011,
generalist R² 0.023, generalist Pearson 0.006, Test 2 AP 0.0002. `generate.py`
is unseeded, so generation variance is included and it dominates the bootstrap.

Consequence: the whole sparse ladder is inseparable on specialist R², and
sparse-vs-low-rank at equal parameter budget — the claim `iht07` was built to
make — is unresolvable at this precision. Prefer generalist R², Pearson, or
Test 2 AP, which are 10–100× tighter.

## Running one

```bash
# Always smoke first. The line to read is each hospital's reported density:
# 1.0000 on the l1 arm means the proximal step never fired.
sbatch examples/fedpyhealth/scripts/sbatch/smoke_sparse_adapters.sbatch

# Then train + score the ladder (reuses the ft_epochs=20 FedAvg trunk).
sbatch examples/fedpyhealth/scripts/sbatch/_run_sparse_adapters.sh
```

Ask `train.py` where a run will land rather than rebuilding the name:

```bash
python examples/fedpyhealth/train.py --profile full --regime fedavg_ft \
    --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" --no-early-stop \
    --ft-epochs 20 --adapter last_mlp_iht --adapter-sparsity 0.1 \
    --print-save-dir
```

## Budgets are not interchangeable

`fed_rounds = (total_epochs − ft_epochs) / local_epochs`, so raising
`--ft-epochs` **shortens the FedAvg trunk**: a 20-epoch adapter starts from a
40-round global, a 2-epoch one from 49. The `--adapter none` run *at the same
ft_epochs* is the only correct full-fine-tune baseline for a given set. Do not
compare across budgets.

## Known caveat

`compute_prevalence_metrics` bootstraps without a seed, so scoring the same
model twice differs by Δ Pearson ≈ 0.019 and Δ R² ≈ 0.079. That is comparable to
the entire spread between adapter variants at 2 epochs. Treat differences below
it as noise until the seed is pinned.

See also [irm-federated.md](irm-federated.md) — the IRM penalty applies to the
FedAvg rounds, never to fine-tuning, so `--irm-rho` on a `fedavg_ft` run means
"IRM trunk, plain local fine-tuning", which composes with any adapter.
