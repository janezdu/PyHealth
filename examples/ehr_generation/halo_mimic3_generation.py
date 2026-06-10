"""Generate synthetic MIMIC-III patients from a trained HALO checkpoint.

Run ``halo_mimic3_training.py`` first to produce the checkpoint. The dataset
and task here must match training exactly so the vocabulary is reconstructed
identically, and the model hyperparameters must match so the checkpoint loads.
"""

import json
import os
import tempfile

import torch

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models import HALO
from pyhealth.tasks import HaloGenerationMIMIC3

# Must match halo_mimic3_training.py.
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 3
N_CTX = 20
SAVE_DIR = "./save/halo_mimic3"

NUM_SAMPLES = 100
OUTPUT = "synthetic_mimic3.json"

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

    # STEP 3: define model and load the trained checkpoint
    model = HALO(
        dataset=sample_dataset,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        save_dir=SAVE_DIR,
    )
    checkpoint = torch.load(os.path.join(SAVE_DIR, "halo_model"), map_location="cpu")
    model.halo_model.load_state_dict(checkpoint["model"])

    # STEP 4: generate synthetic patients
    synthetic = model.synthesize_dataset(num_samples=NUM_SAMPLES)

    # STEP 5: save output
    with open(OUTPUT, "w") as f:
        json.dump(synthetic, f, indent=2)
    print(f"Wrote {len(synthetic)} synthetic patients to {OUTPUT}")
