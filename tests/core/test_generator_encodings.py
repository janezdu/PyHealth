"""Every generator sharing EHRGeneration must read the task's encoding.

HALO, GPT2 and PromptEHR all consume ``EHRGeneration``, so changing that task's
``input_schema`` changes what all three receive. The failure mode this guards
is silent: a multi-hot row read as raw values yields 1.0 for every present
code, and 1 is ``<unk>``, so a model would train happily on nothing but unknown
codes. These tests assert the real codes survive, under either encoding.
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import GPT2, PromptEHR

SAMPLES = [
    {"patient_id": "p0", "visits": [["A05B", "A05C"], ["A11D"], ["C129"]]},
    {"patient_id": "p1", "visits": [["A05B"], ["A04A", "B035"]]},
    {"patient_id": "p2", "visits": [["C129", "A11D"], ["A05C"], ["A04A"]]},
    {"patient_id": "p3", "visits": [["B035"], ["A05B", "C129"]]},
]


def _dataset(schema_key, name):
    return create_sample_dataset(
        samples=SAMPLES,
        input_schema={"visits": schema_key},
        output_schema={},
        dataset_name=name,
    )


class TestVisitCodeIds(unittest.TestCase):
    """Each nested processor inverts its own encoding to the same code ids."""

    def test_both_processors_agree(self):
        multihot = _dataset("nested_multihot", "vci_mh")
        indexed = _dataset("nested_sequence", "vci_ix")
        mh_proc = multihot.input_processors["visits"]
        ix_proc = indexed.input_processors["visits"]

        # Same samples, same traversal order, so the vocabularies must match --
        # that is what makes the per-visit comparison below meaningful.
        self.assertEqual(mh_proc.code_vocab, ix_proc.code_vocab)

        for i in range(len(SAMPLES)):
            mh_row = multihot[i]["visits"]
            ix_row = indexed[i]["visits"]
            for visit in range(mh_row.shape[0]):
                # Multi-hot returns vocabulary order, the index form charted
                # order, so compare as sets.
                self.assertEqual(
                    set(mh_proc.visit_code_ids(mh_row[visit])),
                    set(ix_proc.visit_code_ids(ix_row[visit])),
                )

    def test_multihot_ids_are_not_all_unk(self):
        """The specific regression: reading values instead of column indices."""
        processor = _dataset("nested_multihot", "vci_unk").input_processors["visits"]
        row = processor.process([["A05B", "A05C"]])[0]
        ids = processor.visit_code_ids(row)
        self.assertEqual(len(ids), 2)
        self.assertNotIn(processor.UNK, ids)


class TestGeneratorsAcceptEitherEncoding(unittest.TestCase):
    """GPT2 and PromptEHR serialise real codes from either processor."""

    MODELS = [
        (GPT2, {"embed_dim": 16, "n_heads": 2, "n_layers": 2, "max_len": 64}),
        (PromptEHR, {"embed_dim": 16, "n_heads": 2, "n_layers": 2, "max_len": 64,
                     "prompt_length": 4}),
    ]

    def _streams(self, model, visits):
        if hasattr(model, "_serialize"):
            return model._serialize(visits)
        input_ids, _, _ = model._encode_visits(visits)
        return [row.tolist() for row in input_ids]

    def test_codes_survive_serialisation(self):
        for cls, kwargs in self.MODELS:
            with self.subTest(model=cls.__name__):
                dataset = _dataset("nested_multihot", f"gen_{cls.__name__}")
                model = cls(dataset=dataset, batch_size=2, epochs=1, **kwargs)
                visits = torch.stack([dataset[i]["visits"] for i in range(2)])
                streams = self._streams(model, visits)

                code_ids = [
                    t
                    for stream in streams
                    for t in stream
                    if t < model.code_vocab_size and t != 0
                ]
                self.assertTrue(code_ids, "no code tokens were emitted at all")
                # The bug turned every code into <unk>; a real stream carries
                # several distinct codes.
                self.assertGreater(
                    len(set(code_ids) - {model.visits_processor.UNK}),
                    1,
                    f"{cls.__name__} emitted only <unk>: the visit row was read "
                    "as raw values instead of via visit_code_ids",
                )


if __name__ == "__main__":
    unittest.main()
