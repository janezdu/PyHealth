# ICLR 2027 rebuttal plan — submission 34654 (FedLIBRA)

Scores: RS1V 2 (conf 3, sound 3) · dob9 **1** (conf 3, sound 1) · Zda4 2 (conf 3, sound 2)

Zda4 is the swing vote ("the main reason I would like to see the paper accepted once the
problems are fixed"). dob9 is soundness-1 and will not move without the de-confounded
baseline story. RS1V's objections are a subset of the other two plus the calibration
confound, which is the single most dangerous criticism in the batch.

## The seven complaints, merged across reviewers

| # | Complaint | Who | Kind |
|---|-----------|-----|------|
| C1 | "Frontier" is a line through two points; beating it proves nothing | dob9 W1, Zda4 W1 | framing |
| C2 | Centralized/baseline collapse (R² −53.1/−44.3); the fix confounds rare-upweighting with early stopping; each condition run once | dob9 W2+Q1, RS1V Q2, Zda4 W1+Q2 | **experiment** |
| C3 | IRM gain may be calibration, not invariance | RS1V W1+Q1 | **experiment** |
| C4 | best-of-K alone hurts, text contradicts Table 4, IRM+LoRA baseline missing | all three | **experiment** |
| C5 | No seeds / CIs; gaps may be noise | dob9, Zda4 W2+Q3 | **experiment** |
| C6 | AUROC 0.420 below chance for real local data = looks like a bug | all three | **diagnosis** |
| C7 | Privacy claims too strong; no DP, no attack on shared updates; utility shows no advantage | RS1V W3+W4, Zda4 Q10 | framing + experiment |

## Tier 0 — already in `_outputs`, no GPU, highest leverage

**T0.1 Run-to-run variance (answers C5 immediately).** Three independent trainings of the
same arm already exist: `fedavg_E2_R50_hilo8_random_privacy_irm100w17_nes_do0.1_xm4`,
`varrep2_…`, `varrep3_…`. Specialist rare-code numbers:

| rep | R² | Pearson |
|-----|----|---------|
| 1 | −0.025 | 0.620 |
| 2 | −0.008 | 0.629 |
| 3 | +0.007 | 0.629 |

Training-run SD ≈ **0.016 R²**, **0.005 Pearson** — two orders of magnitude below the
0.426→0.716 gap the paper claims. Caveats to state honestly: dropout-0.1 variant, val fold,
specialist only. Promote to a table; back it with T2.1 reps on the headline arms.

**T0.2 Replace the frontier line with a measured achievable region (C1).** Already have
~40 runs that trace the interior: `fedavg_ft_sweep-000..005` (FT epochs 1–25),
`ftsweep_e{2,10,25}_lr{1e-5,1e-4,3e-4}`, Ditto λ ∈ {0, 1e-3…10}, FedProx μ 5-point sweep,
`latent_width_v1-000..005`, dropout sweep. Scatter all of them in the (gen, spec) plane and
draw the empirical upper-right envelope. Then the claim becomes "FedLIBRA sits outside the
envelope traced by every knob of every baseline we swept," which is defensible, instead of
"it beats a line between two points," which is not. `scripts/fig_specialist_generalist.py`
already does most of this.

**T0.3 De-confound the centralized reference (C2).** The 2×2 exists:
`centralized_full_hilo8_random` (plain), `_rw` (upweight), `_nes` (no early stop),
`_rw_irm100`, `_nrounds{80,100}`. Score all four at b200 into one table with the
weighting and early-stopping effects separated. Pair it with the known mechanism from
`notes/` — codes/visit climbs 4 → 57 with uninterrupted training while val loss stays flat —
so the collapse has a stated cause, not just a patch.

**T0.4 IRM+LoRA without best-of-K (C4, dob9's explicit Q2).** The run exists:
`fedavg_ft_full_hilo8_random_nes_ftepochs20_adapterlora_head_irmrho100_irmwarmup34`.
Needs a pooled rescore at 200 bootstraps, then goes straight into Table 4. Also fix the
ablation prose: best-of-K alone *does* hurt (−1.693/−0.430) and the current text says the
opposite.

**T0.5 K sweep + compute cost (C4).** `do0.3_xm{2,4,8}`, `smoke_xm_k{1,3}`,
`latent_width_v1-*` give K and z-width. Add wall-clock/FLOPs per round from the SLURM logs.

**T0.6 DP row (C7).** `dpfix_s{0.5,1.0,2.0}` + `dp_cost.png` are done (the earlier `dp_s*`
runs are invalid — they noised the attention mask). Put the ε-vs-fidelity curve in the
privacy appendix and soften the abstract.

## Tier 1 — new CPU-only scoring over existing synthetic sets

**T1.1 Hospital discriminator (C3, C4, "did IRM work?").** Train a classifier to predict
which site a synthetic record came from; reference it against the same classifier on *real*
records. If IRM worked, synthetic site-identifiability should fall toward the real-data
floor relative to FedAvg. This is the only direct evidence of invariance in the paper and
two reviewers asked for it by name. New script, no GPU.

**T1.2 Sequence-level fidelity (Zda4 Q5).** Visits/patient and codes/visit distributions,
code co-occurrence (lift or PMI correlation on the top-N pairs), and held-out NLL of each
generator. Currently the paper scores only marginal prevalence, which is dob9's W4. The
codes/visit statistic doubles as the centralized-collapse diagnostic.

**T1.3 Diagnose AUROC 0.420 (C6).** Do this first — all three reviewers flagged it and the
current appendix explanation ("overfitting") does not explain *below chance*, which means
anti-correlated predictions, i.e. a label-orientation or masking bug in
`test2_rare_efficacy.py`. Either find the bug and rerun the utility table, or produce a
positive control (a classifier on real pooled data under the identical pipeline) proving the
pipeline is sound and the number is a small-n artifact. Do not defend 0.420 without one of
those two.

## Tier 2 — new GPU training, in priority order

**T2.1 Temperature-calibration control (C3, RS1V Q1).** Per-hospital sampling temperature
tuned on each site's val fold, applied to FedAvg and centralized, then rescored.
`generate.py --temperature` already exists, so this is regeneration only, no retraining. If
the IRM gain survives, C3 is dead; if it does not, the paper's central mechanism claim has
to change and it is better to know now.

**T2.2 Seed replicates for the four headline arms (C5).** FedLIBRA, FedAvg+FT, FedRep,
Ditto × 3 seeds. Prerequisite: `train.py:74` hardcodes `SEED = 0` — needs a `--seed` flag
threaded through torch/numpy/dataloader shuffling. Report CIs in the main table.

**T2.3 Matched validation-based checkpoint selection for every arm (C2, dob9 Q1).** The
`nes` runs disabled early stopping. Rerun all arms with the same val-based selection so no
arm gets a bespoke stopping rule.

**T2.4 Rare-upweighting for every method (Zda4 Q2).** Upweighting is currently applied only
to the pooled reference, which is exactly the asymmetry Zda4 objects to.

**T2.5 Second hospital subset (RS1V Q3, Zda4 Q9).** `lo8_random`, `strat8_random` and
`pub8` cohorts already have partial run sets — finish one of them rather than building a new
cohort. MIMIC-III is a new dataset, new vocabulary and new tasks: out of scope for a
rebuttal window, and say so plainly rather than promising it.

## Framing changes (no experiments)

- **Abstract/title claim → R²-only (Zda4 Q1).** Pearson is worse than simpler baselines on
  both axes. Conceding this costs little and buys credibility, since Zda4's own second
  "strength" is that the paper exposed the Pearson/R² gap.
- **Drop "bypass legal restrictions"** and anything implying FL alone prevents PHI leakage.
  Replace with: distance-based tests show no detectable memorization; they are not a formal
  guarantee; DP results are in the appendix; attacks on shared updates are untested and
  named as a limitation.
- **Utility: stop claiming an advantage (RS1V W3).** The honest reading is already in the
  paper's own text — TSTR cannot distinguish right rates from wrong rates. Make that the
  finding instead of a win, and point to the FedRep/LG-FedAvg R² collapse.
- **Answer "why should a personalized model also do well pooled?" (RS1V Q5)** in the problem
  setup, not the rebuttal: a site releasing synthetic data for the federation needs one
  artifact that serves both uses.
- **Fix the Table 4 / ablation-text contradiction** (C4) — cheap, and two reviewers cite it
  as evidence of carelessness.

## Open questions for Jane

1. What is the rebuttal deadline and the per-reviewer word/comment limit? It decides how
   much of Tier 2 is reachable.
2. GPU budget on Delta for the window — T2.1–T2.4 is roughly 20–30 A100-runs.
3. Is `paper/` currently the submitted version or already partly revised? `tables/main_results.tex`
   shows centralized at 0.134/0.858, but reviewers quote −53.1/−44.3, so the two have diverged
   and the rebuttal has to be written against what the reviewers actually read.
   Default assumption if unanswered: `paper/` is post-submission, and every number quoted back
   to a reviewer gets checked against the submitted PDF first.
