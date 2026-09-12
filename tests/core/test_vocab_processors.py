import unittest
from typing import List, Dict, Any
from pyhealth.processors import (
    SequenceProcessor,
    StageNetProcessor,
    NestedSequenceProcessor,
    DeepNestedSequenceProcessor,
    NestedMultiHotProcessor,
)

class TestVocabProcessors(unittest.TestCase):
    """
    Test remove and retain methods for processors with vocabulary support.
    covers: SequenceProcessor, StageNetProcessor, NestedSequenceProcessor, DeepNestedSequenceProcessor
    """

    def test_sequence_processor_remove(self):
        processor = SequenceProcessor()
        samples = [
            {"codes": ["A", "B", "C"]},
            {"codes": ["D", "E"]},
        ]
        processor.fit(samples, "codes")
        original_vocab = set(processor.code_vocab.keys())
        self.assertTrue({"A", "B", "C", "D", "E"}.issubset(original_vocab))

        # Remove "A" and "B"
        processor.remove({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        self.assertIn("<unk>", new_vocab)
        self.assertIn("<pad>", new_vocab)
        
        # Verify processing still works (A and B become <unk>)
        res = processor.process(["A", "C"])
        unk_idx = processor.code_vocab["<unk>"]
        c_idx = processor.code_vocab["C"]
        self.assertEqual(res[0].item(), unk_idx)
        self.assertEqual(res[1].item(), c_idx)

    def test_sequence_processor_retain(self):
        processor = SequenceProcessor()
        samples = [
            {"codes": ["A", "B", "C"]},
            {"codes": ["D", "E"]},
        ]
        processor.fit(samples, "codes")
        
        # Retain "A" and "B"
        processor.retain({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)
        self.assertNotIn("E", new_vocab)
        self.assertIn("<unk>", new_vocab)
        self.assertIn("<pad>", new_vocab)

    def test_sequence_processor_add(self):
        processor = SequenceProcessor()
        samples = [
            {"codes": ["A", "B", "C"]},
        ]
        processor.fit(samples, "codes")
        
        processor.add({"D", "E"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        
        # Test processing with new vocabulary
        res = processor.process(["A", "D"])
        d_idx = processor.code_vocab["D"]
        a_idx = processor.code_vocab["A"]
        self.assertEqual(res[0].item(), a_idx)
        self.assertEqual(res[1].item(), d_idx)

    def test_stagenet_processor_remove(self):
        processor = StageNetProcessor()
        # Flat codes
        samples = [
            {"data": ([0.0, 1.0, 2.0], ["A", "B", "C"])},
            {"data": ([0.0, 1.0], ["D", "E"])},
        ]
        processor.fit(samples, "data")
        
        processor.remove({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        
        # Test processing
        time, res = processor.process(([0.0, 1.0], ["A", "C"]))
        unk_idx = processor.code_vocab["<unk>"]
        c_idx = processor.code_vocab["C"]
        self.assertEqual(res[0].item(), unk_idx)
        self.assertEqual(res[1].item(), c_idx)

    def test_stagenet_processor_retain(self):
        processor = StageNetProcessor()
        # Nested codes
        samples = [
            {"data": ([0.0, 1.0], [["A", "B"], ["C"]])},
            {"data": ([0.0], [["D", "E"]])},
        ]
        processor.fit(samples, "data")
        
        processor.retain({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)

    def test_stagenet_processor_add(self):
        processor = StageNetProcessor()
        # Flat codes
        samples = [
            {"data": ([0.0, 1.0, 2.0], ["A", "B", "C"])},
        ]
        processor.fit(samples, "data")
        
        processor.add({"D", "E"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        
        # Test processing with new vocabulary
        time, res = processor.process(([0.0, 1.0], ["A", "D"]))
        d_idx = processor.code_vocab["D"]
        a_idx = processor.code_vocab["A"]
        self.assertEqual(res[0].item(), a_idx)
        self.assertEqual(res[1].item(), d_idx)

    def test_nested_sequence_processor_remove(self):
        processor = NestedSequenceProcessor()
        samples = [
            {"codes": [["A", "B"], ["C", "D"]]},
            {"codes": [["E"]]},
        ]
        processor.fit(samples, "codes")
        
        processor.remove({"A", "B"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        
        res = processor.process([["A", "C"]])
        unk_idx = processor.code_vocab["<unk>"]
        c_idx = processor.code_vocab["C"]
        # res shape (1, max_inner_len)
        # First code in first visit should be unk, second C
        # Note: processor padds to max_inner_len
        visit = res[0]
        self.assertEqual(visit[0].item(), unk_idx)
        self.assertEqual(visit[1].item(), c_idx)

    def test_nested_sequence_processor_retain(self):
        processor = NestedSequenceProcessor()
        samples = [
            {"codes": [["A", "B"], ["C", "D"]]},
            {"codes": [["E"]]},
        ]
        processor.fit(samples, "codes")
        
        processor.retain({"E"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("E", new_vocab)
        self.assertNotIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)

    def test_nested_sequence_processor_add(self):
        processor = NestedSequenceProcessor()
        samples = [
            {"codes": [["A", "B"], ["C", "D"]]},
        ]
        processor.fit(samples, "codes")
        
        processor.add({"E"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)

    def test_deep_nested_sequence_processor_remove(self):
        processor = DeepNestedSequenceProcessor()
        samples = [
            {"codes": [[["A", "B"], ["C"]], [["D"]]]},
        ]
        processor.fit(samples, "codes")
        
        processor.remove({"A"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertNotIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        
        # Test process
        # Input [[[A]]] -> [[[<unk>]]] (padded)
        res = processor.process([[["A"]]])
        unk_idx = processor.code_vocab["<unk>"]
        # res shape (1, max_visits, max_codes)
        # first group, first visit, first code
        self.assertEqual(res[0, 0, 0].item(), unk_idx)

    def test_deep_nested_sequence_processor_retain(self):
        processor = DeepNestedSequenceProcessor()
        samples = [
            {"codes": [[["A", "B"], ["C"]], [["D"]]]},
        ]
        processor.fit(samples, "codes")
        
        processor.retain({"A"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertNotIn("B", new_vocab)
        self.assertNotIn("C", new_vocab)
        self.assertNotIn("D", new_vocab)

    def test_deep_nested_sequence_processor_add(self):
        processor = DeepNestedSequenceProcessor()
        samples = [
            {"codes": [[["A", "B"], ["C"]], [["D"]]]},
        ]
        processor.fit(samples, "codes")
        
        processor.add({"E"})
        new_vocab = set(processor.code_vocab.keys())
        self.assertIn("A", new_vocab)
        self.assertIn("B", new_vocab)
        self.assertIn("C", new_vocab)
        self.assertIn("D", new_vocab)
        self.assertIn("E", new_vocab)
        
        # Test process
        res = processor.process([[["E"]]])
        e_idx = processor.code_vocab["E"]
        self.assertEqual(res[0, 0, 0].item(), e_idx)

class TestSharedCodeVocabulary(unittest.TestCase):
    """CodeVocabularyMixin gives every code processor the same vocabulary.

    Before the mixin, five processors carried verbatim copies of these methods
    and every copy had the same defect: a removal renumbered the vocabulary but
    left ``_next_index`` untouched, so the next ``fit`` allocated past the end
    of it. Sharing one implementation fixes all five at once, which is what
    these tests pin.
    """

    # Each processor takes a different nesting depth, so the same code list is
    # wrapped to match.
    PROCESSORS = [
        (SequenceProcessor, lambda codes: codes),
        (NestedSequenceProcessor, lambda codes: [codes]),
        (DeepNestedSequenceProcessor, lambda codes: [[codes]]),
        (NestedMultiHotProcessor, lambda codes: [codes]),
    ]

    def test_fit_after_remove_stays_in_range(self):
        """The bug: ``fit`` allocates from ``_next_index``, ``remove`` renumbers.

        Removing codes renumbered the vocabulary to 0..n-1 but left
        ``_next_index`` at its pre-removal value, so the next ``fit`` handed
        out an index past the end of the vocabulary. Anything sizing an
        embedding table by ``vocab_size()`` would then index out of bounds.
        """
        for cls, wrap in self.PROCESSORS:
            with self.subTest(processor=cls.__name__):
                processor = cls()
                processor.fit([{"codes": wrap(["A", "B", "C"])}], "codes")
                processor.remove({"A", "B"})
                processor.fit([{"codes": wrap(["D"])}], "codes")

                indices = list(processor.code_vocab.values())
                self.assertEqual(
                    max(indices), processor.vocab_size() - 1,
                    f"{cls.__name__} allocated an index past the vocabulary: "
                    f"{processor.code_vocab}",
                )
                self.assertEqual(len(indices), len(set(indices)))

    def test_indices_stay_contiguous_from_zero(self):
        for cls, _ in self.PROCESSORS:
            with self.subTest(processor=cls.__name__):
                processor = cls()
                processor.add({"A", "B", "C"})
                processor.retain({"B"})
                self.assertEqual(
                    sorted(processor.code_vocab.values()),
                    list(range(processor.vocab_size())),
                )

    def test_special_tokens_survive_every_edit(self):
        for cls, _ in self.PROCESSORS:
            with self.subTest(processor=cls.__name__):
                processor = cls()
                processor.add({"A"})
                processor.retain(set())
                self.assertEqual(processor.code_vocab["<pad>"], processor.PAD)
                self.assertEqual(processor.code_vocab["<unk>"], processor.UNK)


if __name__ == "__main__":
    unittest.main()
