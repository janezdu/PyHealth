"""Train HALO on MIMIC-III for synthetic EHR generation.

HALO is an autoregressive transformer that learns to generate synthetic
patient visit sequences. Unlike supervised models, it is generative, so it
uses its own training loop instead of ``pyhealth.trainer.Trainer``.

Run ``halo_mimic3_generation.py`` afterwards to sample synthetic patients
from the checkpoint produced here.
"""

import tempfile

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.models import HALO
from pyhealth.tasks import HaloGenerationMIMIC3

# Architecture hyperparameters. Must match halo_mimic3_generation.py so the
# saved checkpoint can be reloaded. Kept small here for a fast demo run;
# scale embed_dim / n_layers / n_ctx / epochs up for real datasets.
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 3
N_CTX = 20
SAVE_DIR = "./save/halo_mimic3"

if __name__ == "__main__":
    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["diagnoses_icd"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: set task
    task = HaloGenerationMIMIC3()
    sample_dataset = base_dataset.set_task(task)

    train_dataset, val_dataset, _ = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    # STEP 3: define model
    # HALO derives its vocabulary size automatically from sample_dataset.
    model = HALO(
        dataset=sample_dataset,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        batch_size=16,
        epochs=5,
        save_dir=SAVE_DIR,
    )

    # STEP 4: train
    # train_model validates after every epoch and saves the best checkpoint
    # to SAVE_DIR/halo_model.
    model.train_model(train_dataset, val_dataset)
