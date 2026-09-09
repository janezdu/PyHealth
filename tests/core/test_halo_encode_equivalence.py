"""HALO's vectorised encoder must match the loop it replaced, exactly.

``_encode_visits`` used to walk (patient, visit, code slot) in Python over the
padded index tensor. On an A100 that cost ~108 ms per patient -- about 99.8% of
a training step, against ~0.03 s for the transformer's own forward and backward.
It is now a vectorised placement over per-visit multi-hot rows.

That is a pure performance change, so the tensors handed to the transformer must
be bit-identical: anything else silently invalidates every existing checkpoint
and every published number. The legacy implementation is reproduced here rather
than imported, so this test keeps its meaning after the original is deleted.
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import HALO
from pyhealth.processors import NestedSequenceProcessor


def legacy_encode(cfg, visits, device):
    """The pre-vectorisation implementation, verbatim, over index input."""
    batch_size = visits.shape[0]
    batch_ehr = torch.zeros(batch_size, cfg.n_ctx, cfg.total_vocab_size,
                            device=device)
    batch_mask = torch.zeros(batch_size, cfg.n_ctx, 1, device=device)
    start_idx = cfg.code_vocab_size + cfg.label_vocab_size
    end_idx, pad_idx = start_idx + 1, start_idx + 2

    for i in range(batch_size):
        n_visits = int((visits[i].sum(dim=-1) > 0).sum().item())
        n_visits = min(n_visits, cfg.n_ctx - 2)
        for j in range(n_visits):
            for code_idx in visits[i, j]:
                if code_idx > 0:
                    batch_ehr[i, j + 2, code_idx] = 1
            batch_mask[i, j + 2] = 1
        batch_ehr[i, 0, start_idx] = 1
        batch_ehr[i, n_visits + 1, end_idx] = 1
        batch_ehr[i, n_visits + 2:, pad_idx] = 1

    return batch_ehr, batch_mask[:, 1:, :]


class TestHALOEncodeEquivalence(unittest.TestCase):
    """Both encodings of the same patients must produce the same tensors."""

    SAMPLES = [
        {"patient_id": "p0", "visits": [["A05B", "A05C"], ["A11D"], ["C129"]]},
        {"patient_id": "p1", "visits": [["A05B"], ["A04A", "B035"]]},
        {"patient_id": "p2", "visits": [["C129", "A11D"], ["A05C"], ["A04A"]]},
        {"patient_id": "p3", "visits": [["B035"]]},
        # Repeated codes: the index form stores five entries, the multi-hot form
        # one bit. HALO collapsed them either way, which is why this matches.
        {"patient_id": "p4", "visits": [["A05B", "A05B", "A05B"], ["C129"]]},
    ]

    def _datasets(self):
        multihot = create_sample_dataset(
            samples=self.SAMPLES, input_schema={"visits": "nested_multihot"},
            output_schema={}, dataset_name="mh")
        index = create_sample_dataset(
            samples=self.SAMPLES, input_schema={"visits": "nested_sequence"},
            output_schema={}, dataset_name="idx")
        return multihot, index

    def test_bit_identical(self):
        multihot, index = self._datasets()
        model = HALO(dataset=multihot, embed_dim=16, n_heads=2, n_layers=2,
                     n_ctx=8, batch_size=len(self.SAMPLES), epochs=1)

        mh_batch = next(iter(get_dataloader(multihot, batch_size=len(self.SAMPLES))))
        ix_batch = next(iter(get_dataloader(index, batch_size=len(self.SAMPLES))))

        new_ehr, new_mask = model._encode_visits(mh_batch["visits"])
        old_ehr, old_mask = legacy_encode(model.config, ix_batch["visits"],
                                          model.device)

        self.assertTrue(torch.equal(new_ehr, old_ehr),
                        f"{(new_ehr != old_ehr).sum().item()} cells differ")
        self.assertTrue(torch.equal(new_mask, old_mask))

    def test_truncates_past_context(self):
        """A patient with more visits than n_ctx-2 is cut, not wrapped."""
        many = [{"patient_id": "long", "visits": [["A05B"]] * 20}]
        ds = create_sample_dataset(samples=many + self.SAMPLES,
                                   input_schema={"visits": "nested_multihot"},
                                   output_schema={}, dataset_name="long")
        model = HALO(dataset=ds, embed_dim=16, n_heads=2, n_layers=2, n_ctx=8,
                     batch_size=2, epochs=1)
        batch = next(iter(get_dataloader(ds, batch_size=2)))
        ehr, mask = model._encode_visits(batch["visits"])
        self.assertEqual(ehr.shape[1], 8)
        self.assertEqual(mask.shape[1], 7)

    def test_no_inner_padding_in_multihot(self):
        """The multi-hot row width is the vocabulary, not the longest visit.

        This is the memory win: on eICU the index form pads every visit to 3,951
        slots because one visit somewhere is that long.
        """
        multihot, index = self._datasets()
        mh_w = multihot.input_processors["visits"].vocab_size()
        ix_w = index.input_processors["visits"]._max_inner_len
        self.assertEqual(next(iter(multihot))["visits"].shape[1], mh_w)
        self.assertEqual(next(iter(index))["visits"].shape[1], ix_w)


if __name__ == "__main__":
    unittest.main()
