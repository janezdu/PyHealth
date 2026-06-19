# Rule: the repo map and the pipeline contract

PyHealth is one pipeline. Almost everything users do is **slotting a new piece into an
existing stage** — not inventing new machinery. Orient by the stage, then find the folder.

```
   Dataset            Task                Model            Trainer          Metrics
 (load raw      →  (define the      →  (the neural   →  (train/eval   →  (score the
  records)          ML problem)         network)         loop)            predictions)
```

| Stage | Lives in | You extend it by | Key contract |
|-------|----------|------------------|--------------|
| **Dataset** | `pyhealth/datasets/` (+ YAML in `configs/`) | adding `yourdata.py` + `configs/yourdata.yaml` | subclass `BaseDataset`; load tables into a polars frame |
| **Task** | `pyhealth/tasks/` | adding `your_task.py` | subclass `BaseTask`; set `input_schema`/`output_schema`; `__call__(patient) -> List[Dict]` of samples |
| **Model** | `pyhealth/models/` (~45 exist) | adding `your_model.py` | subclass `BaseModel`; `forward(...)` returns a dict incl. `loss` and `y_prob`/`y_true` |
| **Trainer** | `pyhealth/trainer.py` | usually *not* extended — you use it | `Trainer(model).train(train_dataloader, val_dataloader, epochs, monitor=...)`; `.evaluate(loader) -> Dict[str, float]` |
| **Metrics** | `pyhealth/metrics/` | adding a metric fn | takes predictions, returns scores; see subpackages (e.g. `generative/`) |

## How the stages connect (the canonical flow)

```python
from pyhealth.datasets import MIMIC4Dataset, split_by_patient, get_dataloader
from pyhealth.tasks import <SomeTask>
from pyhealth.models import <SomeModel>
from pyhealth.trainer import Trainer

ds = MIMIC4Dataset(root="...", tables=[...], dev=True)   # 1. raw records (dev = small)
samples = ds.set_task(<SomeTask>())                       # 2. -> SampleDataset of samples
tr, va, te = split_by_patient(samples, [0.8, 0.1, 0.1])   # 3. patient-level split (no leakage)
model = <SomeModel>(dataset=samples, ...)                 # 4. model knows the schema
Trainer(model).train(get_dataloader(tr, batch_size=32, shuffle=True),
                     get_dataloader(va, batch_size=32))   # 5. train
metrics = Trainer(model).evaluate(get_dataloader(te, batch_size=32))   # 6. score
```

`set_task` fits processors **once** and returns a `SampleDataset` with a shared vocabulary,
which is what makes splits and models compatible. Split **by patient** (`split_by_patient`),
not by sample, to avoid the same patient leaking across train/test.

## Side modules (not in the main pipeline, but commonly used)

- `pyhealth/medcode/` — medical coding systems and cross-mappings (ICD ↔ CCS, ATC, etc.).
- `pyhealth/calib/` — model calibration / uncertainty.
- `pyhealth/interpret/` — interpretability.
- `pyhealth/graph/`, `pyhealth/nlp/` — graph and clinical-text models.
- `pyhealth/processors/`, `pyhealth/tokenizer.py` — how raw fields become model tensors.

## Practical rule for agents

Before calling any class, **open it and read the actual signature** in its folder — PyHealth
evolves and a remembered signature may be stale. The stage map above tells you *where* to
look; the source is the source of truth for *how* to call it.
