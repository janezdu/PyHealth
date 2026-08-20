# TODO

Outstanding work on the federated synthetic-EHR experiment, roughly in the order
it should be done. Items carry the evidence that motivated them so the reason
survives longer than the conversation.

## A. Stage 1 — federated generator training

### A1. Validation curves (blocks A2)
No validation loss is recorded anywhere today. Every `train_model` call in
`train.py` passes `val_dataset=None`, so the only curves that exist are training
loss — which cannot show overfitting and cannot drive early stopping.

**Do NOT fix this by passing `val_dataset` through.** `HALO.train_model` writes a
"best model" checkpoint to `self.save_dir/halo_model` whenever val loss improves,
and *loads* that file at the top of every call. All 8 FedAvg clients share one
`save_dir`, so enabling validation would make client B silently resume from
client A's weights instead of the broadcast global. FedAvg would stop being
FedAvg, with no error raised.

Instead: compute val loss in `train.py` at round/epoch boundaries and log it as
`loss_val/hospital_<id>`. Each hospital's val split is *already loaded* as
`client_tests` (`EVAL_FOLD = "val"`), so no new data plumbing is needed.

Cost: one forward pass over 827 val patients per epoch, ~14% of a train pass.

### A2. Early stopping for the federated run
Depends on A1 — stopping on training loss is close to useless.

In FedAvg the unit is the **round**, not the epoch: after each aggregation,
score the global model on val, stop when it has not improved for `patience`
rounds, and keep the best global state rather than the last. Then raise the
`n_rounds` default and let patience decide where to stop, instead of hard-coding
50 rounds x 2 local epochs = 100 passes.

### A3. GPU throughput
Measured from SLURM accounting on the finished runs:

| | value |
|---|---|
| GPU utilisation, 12-19h generator runs | 22-26% |
| GPU memory | ~1270 MB of 40960 MB (3%) |
| Host RAM | 2.9 GB used vs 128 GB requested |
| CPU efficiency | 6.3% |

The card idles ~3/4 of the time, so wall-clock is set by data loading and
per-sample Python work rather than by the model. In rough order of expected
payoff:

1. **`batch_size` 64 -> 256/512.** Memory is 3% used. **Caveat:** hospital 438
   holds 79 records, so batch 512 gives it ONE gradient step per epoch. Cap per
   client, e.g. `min(batch_size, max(8, n_train // 4))`.
2. **Mixed precision** (`torch.autocast` + `GradScaler`) around the HALO forward.
3. **DataLoader** `num_workers > 0`, `pin_memory=True`, `persistent_workers=True`
   -- at 22% utilisation the loader is the prime suspect.
4. `_encode_visits` runs per batch in Python; worth profiling before assuming
   the loader is at fault.

Lower the `--mem` request too: 128 GB booked, 2.9 GB used.

### A4. Keep GPU monitoring cheap
`log_gpu()` currently shells out to `nvidia-smi` once per round/epoch (~50-100 ms).
For FedAvg that is 49 calls (~5 s over 18 h); for `local` it is 800 calls (~80 s),
still under 0.2%. Add a `--gpu-sample-every N` knob so it can be turned down or
off, and never sample inside the batch loop.

Note: `nvidia-smi` reports utilisation for the whole card, not this process --
on a shared node the number includes other jobs.

### A5. Visualiser
Once A1 lands, overlay train vs val per series in `viz/loss_curves.html` and mark
where they separate. That divergence point is the "sweet spot" epoch. The page
and its template already exist; only the val series is missing.

## B. Stage 2 — downstream classifiers

### B1. Convergence (biases the headline result)
At `--epochs 10` the synthetic arms have converged (last-3-epoch loss drop ~1%)
but the real baselines have NOT (`real_local` 16.7%, `real_pooled_budgeted`
11.4%). The baselines are the bar the synthetic arms must clear, so the
comparison currently favours synthetic. Set `monitor="pr_auc_macro"`,
`load_best_model_at_last=True`, and raise `--epochs`.

### B2. Normalised per-code AP
Macro-AP averages codes whose floors differ by ~50x, so the mean is dominated by
the least-rare codes. Divide each code's AP by its own prevalence before
averaging.

### B3. The inverted AUROC, still unexplained
Every arm scores below 0.500. Masking has been tested twice and refuted
(mask 119 -> 30 codes moved nothing; burden/length correlation stays +0.75 after
masking). Untested, in priority order:
1. **Length-only baseline** -- rank by post-mask input length alone. The data says
   it should clear 0.5; every trained arm sits at 0.32. If the trivial rule wins,
   the fault is in the model/scoring path, not the data.
2. Label-axis alignment: `label_proc.label_vocab` order vs model output columns.
3. Train on `real_pooled`, score on its own training data -- should be ~1.0.

### B4. `recall_at_k` for the `prior` arm
Not computed in-file, so recall@k has no floor to be read against.

## C. Cohort

### C1. ICD-9 / ICD-10 mixing
44 of 641 rare codes are ICD-10 shadows of conditions that are common under
ICD-9 (`I50.9` 0.048% vs `428.0` 6.544%). Predicting them means learning "this
record was coded in ICD-10" -- a site/era artifact. Four are inside the current
evaluation draws. Decide: collapse synonyms via `pyhealth/medcode/cross_map.py`,
or exclude ICD-10 from the scoreable pool. Either needs a cohort rebuild.

## D. Housekeeping

- `fedavg-l` is `fedavg_ft` (49 rounds + 2 local epochs, unaveraged). Relabel or
  leave -- decision outstanding.
- `viz/index.html` is the stale pre-artifact fidelity-vs-utility page; the
  updated version's source was lost with the session scratchpad and needs
  rebuilding into the repo.
- Nothing is committed.
