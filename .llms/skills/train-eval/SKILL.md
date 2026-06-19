---
name: train-eval
description: Run the core PyHealth loop end-to-end — load a dataset, set a prediction task, train a model, and evaluate it — starting on small demo/dev data. Use when someone wants to "try PyHealth", "predict X from Y", get a first result, or train/evaluate a model on a dataset. Clinician-first: translates the goal into pipeline pieces, runs safe and small, and sanity-checks the result.
---

# train-eval

The most common thing anyone wants from PyHealth: go from *"I want to predict X from Y"* to
*a trained model and a believable score*. This skill drives the whole pipeline on **small,
local data first**, then scales only on request.

Read `.llms/principles.md` first — especially calibrate-to-the-person, translate-both-ways,
default-small-local, and guard-against-silent-wrongness. They govern every step here.

## Step 0 — Turn the goal into pipeline pieces (translate)

Before any code, map the user's plain-language goal onto the four stages
(`rules/00-repo-map.md`). Say the mapping back to them in their language so they can confirm:

> "Predict 30-day readmission from labs and diagnoses" →
> **Dataset** `MIMIC4Dataset` (tables: labs, diagnoses) → **Task** a readmission task →
> **Model** a sequence model (e.g. `Transformer`/`RNN`) → **Metric** AUROC/AUPRC.

If a piece doesn't exist yet (no task for their question, no loader for their data), that's a
`scaffold-task` / `scaffold-dataset` job — say so instead of forcing a bad fit.

## Step 1 — Load the dataset, small (default safe)

Always start with `dev=True` (or a demo subset) and a CPU-friendly size. Confirm the data
location comes from the user via `root=`; never hardcode a path.

```python
from pyhealth.datasets import MIMIC4Dataset
ds = MIMIC4Dataset(root="<user-provided>", tables=["diagnoses_icd", "labevents"], dev=True)
```

**Check before moving on:** the dataset is non-empty and has the tables you expect. If `root`
is wrong, PyHealth can't find the CSVs — explain that plainly and offer the demo path.

## Step 2 — Set the task → SampleDataset

```python
from pyhealth.tasks import <TheTask>
samples = ds.set_task(<TheTask>())
print(len(samples), "samples")          # sanity: > 0, and roughly what you'd expect
print(samples[0])                        # eyeball one sample: right fields, plausible values
```

`set_task` fits processors once and yields a shared vocabulary. **Check:** non-zero samples,
and the sample dict matches the task's `output_schema` (the label is present and not constant).

## Step 3 — Split by PATIENT (no leakage)

```python
from pyhealth.datasets import split_by_patient, get_dataloader
train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1])
train = get_dataloader(train_ds, batch_size=32, shuffle=True)
val   = get_dataloader(val_ds,   batch_size=32, shuffle=False)
test  = get_dataloader(test_ds,  batch_size=32, shuffle=False)
```

Split **by patient**, never by sample — the same patient in both train and test inflates the
score. This is the single most common silent mistake; enforce it.

## Step 4 — Build the model (it reads the schema from the data)

```python
from pyhealth.models import Transformer      # confirm the exact ctor in pyhealth/models/
model = Transformer(dataset=samples)         # model infers feature/label dims from samples
```

Open the model class and read its `__init__` before calling — signatures vary by model.

## Step 5 — Train, small

```python
from pyhealth.trainer import Trainer
trainer = Trainer(model=model)
trainer.train(train_dataloader=train, val_dataloader=val, epochs=3, monitor="roc_auc")
```

Keep `epochs` tiny for the first pass (minutes, CPU/1-GPU). **Check:** loss actually moves;
the run saw > 0 batches. A flat loss or "0 batches" means an upstream problem (empty split,
bad schema) — stop and trace it, don't proceed to a meaningless score.

## Step 6 — Evaluate, then sanity-check the number

```python
results = trainer.evaluate(test)     # -> {"roc_auc": ..., "pr_auc": ..., ...}
print(results)
```

Before reporting, **interrogate the score** (principle 4):
- AUROC pinned at exactly 0.5 → model learned nothing / labels shuffled.
- AUROC ~1.0 on a hard clinical task → suspect leakage (split by sample? a feature that
  encodes the label? test overlapping train?).
- NaN / 0.0 → empty or degenerate test set.

Then **translate the result** (principle 2): say what the metric means in plain terms, and
whether it's trustworthy — not just the digits.

## Step 7 — Offer the next rung (ramp)

If it worked and the user might go further:
- Scale up *on request*: `dev=False`, more epochs, GPU. Flag that real data may need access
  and more compute (and the `slurm-delta` skill if they have HPC).
- If they built something reusable (a new task/dataset framing), offer to package and
  contribute it back via `scaffold-task` / `scaffold-dataset`.

## Footguns checklist (the things that silently ruin a run)

- [ ] Split by **patient**, not sample.
- [ ] Samples non-empty; label present and not constant.
- [ ] Started on `dev=True` / demo data, CPU-friendly size.
- [ ] Loss moved during training; > 0 batches seen.
- [ ] Score sanity-checked (not 0.5 / not suspiciously 1.0 / not NaN) before it's reported.
- [ ] No hardcoded data paths; `root` came from the user.
