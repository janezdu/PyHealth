"""EHR sequence-generation tasks for PyHealth generative models.

These back every generator in :mod:`pyhealth.models.generators`. They extract,
for each patient, the ordered list of visits where each visit is the list of
medical codes recorded in that admission. There is no prediction label, so
``output_schema`` is empty.

Extraction is shared -- :class:`EHRGeneration` holds all of it -- but the
encoding is not, because each generator family reads its codes in a different
shape:

- :class:`VisitMultiHotGeneration` -- one multi-hot row per visit, for HALO,
  whose transformer consumes multi-hot vectors directly.
- :class:`VisitSequenceGeneration` -- per-visit code indices, for the token
  models GPT2 and PromptEHR, which flatten visits into streams of code ids.
- :class:`PatientCodeSetGeneration` -- one pooled code set per patient, for the
  bag-of-codes models MedGAN and CorGAN, which have no visit axis.

Match the task to the model. Handing a model the wrong encoding does not raise:
the numbers still have the right shape and dtype, so training runs and produces
a confidently wrong result. That is why these are separate classes rather than
one task with a flag.

``event_type`` / ``code_attr`` select the dataset's coding columns and can be
passed to any of them, so the encoding and the dataset are independent choices.

Evaluating generated data
-------------------------
The privacy/utility metrics in :mod:`pyhealth.metrics.generative` (``utils.py``,
``privacy.py``, ``utility.py`` -- exposed through ``evaluate_synthetic_ehr``)
consume **long-form** dataframes: one row per ``(patient, visit, code)`` with
columns ``id`` / ``time`` / ``visit_codes`` / ``labels``. ``id`` is the patient
identifier, ``time`` the (integer) visit index, ``visit_codes`` a single code
string, and ``labels`` a patient-level binary label (reduced via ``max`` over
the patient's rows).

Both the real task samples and a generator's ``generate()`` output use the same
``{"visits": [[code, ...], ...]}`` record shape, so
:func:`to_evaluation_dataframe` converts either into that long-form table. A
processed ``SampleDataset`` can be turned back into records with
:func:`decode_dataset`. Subjects are renumbered sequentially (0, 1, 2, ...) in
the ``id`` column -- synthetic patients do not correspond to real ones, so any
original ``patient_id`` is ignored.

.. code-block:: python

    from pyhealth.tasks.generate_ehr import decode_dataset, to_evaluation_dataframe
    from pyhealth.metrics.generative import evaluate_synthetic_ehr

    # Real train/test EHR come from the processed SampleDataset(s):
    train_df = to_evaluation_dataframe(decode_dataset(train_dataset))
    test_df = to_evaluation_dataframe(decode_dataset(test_dataset))

    # Synthetic EHR comes straight from the trained generator (HALO, GPT2, ...):
    synthetic = model.generate(num_samples=len(train_dataset))
    syn_df = to_evaluation_dataframe(synthetic)

    # Privacy metrics need no labels:
    results = evaluate_synthetic_ehr(train_df, test_df, syn_df, metrics="privacy")

The **utility** metrics (machine-learning efficacy, next-visit prediction)
additionally require a meaningful binary ``labels`` column. Since this task is
unconditional (no labels), pass a ``label_fn`` to derive one per patient -- e.g.
``label_fn=lambda r: any("250" in c for v in r["visits"] for c in v)`` for a
diabetes flag -- and the same ``label_fn`` must be applied to the real and
synthetic frames. With no label available, restrict to ``metrics="privacy"``.

Note:
    The MLE component currently hard-codes the downstream task to
    next-visit prediction, which is degenerate for bag-of-codes
    generators (MedGAN, CorGAN) that emit a single aggregate visit per
    patient. A future revision will let callers plug in static-label
    tasks (e.g. mortality, readmission, "ever diagnosed with X") so MLE
    is meaningful for both sequential (HALO, GPT2, PromptEHR) and
    bag-of-codes generators. Until then, restrict bag-of-codes
    evaluation to ``metrics="privacy"`` plus the prevalence metrics.
"""

import logging
from collections.abc import Callable
from typing import ClassVar

from pyhealth.data.data import Patient
from pyhealth.processors import (
    MultiHotProcessor,
    NestedMultiHotProcessor,
    NestedSequenceProcessor,
)

from .base_task import BaseTask

logger = logging.getLogger(__name__)


class EHRGeneration(BaseTask):
    """Per-visit code extraction for unconditional EHR generators.

    Builds one sample per qualifying patient: the ordered list of visits, each
    visit being the list of codes (read from ``code_attr`` on ``event_type``
    events) recorded in that admission. Patients with fewer than ``min_visits``
    qualifying visits are skipped.

    **This class does not set an** ``input_schema`` **and cannot be used
    directly.** Extraction is shared, but each generator family wants the codes
    in a different shape, and handing a model the wrong one fails silently
    rather than loudly. Pick the subclass that matches your model:

    ==============================  =====================  ==================
    Task                            Encoding               Models
    ==============================  =====================  ==================
    :class:`VisitMultiHotGeneration`  per-visit multi-hot  HALO
    :class:`VisitSequenceGeneration`  per-visit indices    GPT2, PromptEHR
    :class:`PatientCodeSetGeneration` one set per patient  MedGAN, CorGAN
    ==============================  =====================  ==================

    Args:
        code_mapping: Optional vocabulary mapping, see :class:`BaseTask`.
        event_type: Event type to pull per admission. Defaults to the class
            attribute (``"diagnoses_icd"``).
        code_attr: Event attribute holding the code string. Defaults to the
            class attribute (``"icd9_code"``).
        min_visits: Minimum qualifying visits to keep a patient. Defaults to
            the class attribute (2).

    Examples:
        >>> from pyhealth.tasks import VisitSequenceGeneration
        >>> task = VisitSequenceGeneration(code_attr="icd_code")
        >>> task.code_attr
        'icd_code'
    """

    task_name: str = "ehr_generation"
    output_schema: ClassVar[dict[str, str | type]] = {}

    event_type: str = "diagnoses_icd"
    code_attr: str = "icd9_code"
    min_visits: int = 2

    def __init__(
        self,
        code_mapping=None,
        event_type: str | None = None,
        code_attr: str | None = None,
        min_visits: int | None = None,
    ) -> None:
        if not hasattr(type(self), "input_schema"):
            raise TypeError(
                f"{type(self).__name__} does not declare an encoding. "
                "EHRGeneration only holds the shared extraction logic -- use "
                "VisitMultiHotGeneration (HALO), VisitSequenceGeneration "
                "(GPT2, PromptEHR) or PatientCodeSetGeneration (MedGAN, "
                "CorGAN), whichever matches your model."
            )
        super().__init__(code_mapping=code_mapping)
        # Per-instance overrides, so a dataset preset is a constructor argument
        # rather than yet another subclass in the encoding x dataset grid.
        if event_type is not None:
            self.event_type = event_type
        if code_attr is not None:
            self.code_attr = code_attr
        if min_visits is not None:
            self.min_visits = min_visits

    def _visits(self, patient: Patient) -> list[list[str]]:
        """Ordered per-admission code lists, empty admissions dropped."""
        visits: list[list[str]] = []
        for admission in patient.get_events(event_type="admissions"):
            events = patient.get_events(
                event_type=self.event_type,
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            codes = [
                getattr(event, self.code_attr)
                for event in events
                if getattr(event, self.code_attr, None)
            ]
            if codes:
                visits.append(codes)
        return visits

    def __call__(self, patient: Patient) -> list[dict]:
        """Extract the per-visit code sequence for a patient."""
        visits = self._visits(patient)
        if len(visits) < self.min_visits:
            return []
        return [{"patient_id": patient.patient_id, "visits": visits}]


class VisitMultiHotGeneration(EHRGeneration):
    """Per-visit code sets as multi-hot rows. For HALO.

    HALO's transformer consumes a multi-hot vector per context position, so
    this hands it exactly that and no repacking happens on the way in.

    Examples:
        >>> from pyhealth.tasks import VisitMultiHotGeneration
        >>> samples = dataset.set_task(VisitMultiHotGeneration())
        >>> samples[0]["visits"].shape  # (num_visits, vocab_size)
        torch.Size([3, 512])
    """

    task_name: str = "ehr_generation_visit_multihot"
    input_schema: ClassVar[dict[str, str | type]] = {
        "visits": NestedMultiHotProcessor
    }


class VisitSequenceGeneration(EHRGeneration):
    """Per-visit code indices, right-padded. For GPT2 and PromptEHR.

    Both are token-sequence models: they flatten each visit into a stream of
    code ids. Indices are what they need, so this avoids encoding to multi-hot
    and decoding straight back.

    Examples:
        >>> from pyhealth.tasks import VisitSequenceGeneration
        >>> samples = dataset.set_task(VisitSequenceGeneration())
        >>> samples[0]["visits"].shape  # (num_visits, max_codes_per_visit)
        torch.Size([3, 12])
    """

    task_name: str = "ehr_generation_visit_sequence"
    input_schema: ClassVar[dict[str, str | type]] = {
        "visits": NestedSequenceProcessor
    }


class PatientCodeSetGeneration(EHRGeneration):
    """One code set per patient, visit structure discarded. For MedGAN/CorGAN.

    Bag-of-codes generators emit a single aggregate vector per patient, so the
    visit axis is collapsed here rather than inside the model. ``min_visits``
    still applies -- it filters on the patient's real visit count before the
    codes are pooled.

    Note:
        Because the visit axis is gone, the next-visit utility metric in
        :mod:`pyhealth.metrics.generative` is not meaningful for these models;
        see this module's header.

    Examples:
        >>> from pyhealth.tasks import PatientCodeSetGeneration
        >>> samples = dataset.set_task(PatientCodeSetGeneration())
        >>> samples[0]["visits"].shape  # (vocab_size,)
        torch.Size([512])
    """

    task_name: str = "ehr_generation_patient_codeset"
    input_schema: ClassVar[dict[str, str | type]] = {"visits": MultiHotProcessor}

    def __call__(self, patient: Patient) -> list[dict]:
        """Pool every visit's codes into one per-patient set."""
        visits = self._visits(patient)
        if len(visits) < self.min_visits:
            return []
        codes = sorted({code for visit in visits for code in visit})
        return [{"patient_id": patient.patient_id, "visits": codes}]


class EHRGenerationMIMIC3(VisitMultiHotGeneration):
    """EHR generation task for MIMIC-III (ICD-9 diagnosis codes), for HALO.

    A :class:`VisitMultiHotGeneration` preset. For GPT2/PromptEHR on MIMIC-III
    use ``VisitSequenceGeneration()``, whose defaults are already MIMIC-III's.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import EHRGenerationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> samples = dataset.set_task(EHRGenerationMIMIC3())
    """

    task_name: str = "ehr_generation_mimic3"
    event_type: str = "diagnoses_icd"
    code_attr: str = "icd9_code"


class EHRGenerationMIMIC4(VisitMultiHotGeneration):
    """EHR generation task for MIMIC-IV (ICD diagnosis codes), for HALO.

    A :class:`VisitMultiHotGeneration` preset. For another encoding on MIMIC-IV
    pass the same columns, e.g. ``VisitSequenceGeneration(code_attr="icd_code")``.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import EHRGenerationMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimiciv/2.2/",
        ...     ehr_tables=["patients", "admissions", "diagnoses_icd"],
        ... )
        >>> samples = dataset.set_task(EHRGenerationMIMIC4())
    """

    task_name: str = "ehr_generation_mimic4"
    event_type: str = "diagnoses_icd"
    code_attr: str = "icd_code"


# ----------------------------------------------------------------------------
# Conversion helpers for pyhealth.metrics.generative.evaluate_synthetic_ehr
# ----------------------------------------------------------------------------
def to_evaluation_dataframe(
    records,
    label_fn: Callable[[dict], int] | None = None,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
):
    """Flatten EHR-generation records into the long-form evaluation dataframe.

    Produces the one-row-per-``(patient, visit, code)`` table consumed by
    :func:`pyhealth.metrics.generative.evaluate_synthetic_ehr` (and the
    ``utils.py`` / ``privacy.py`` / ``utility.py`` functions beneath it).

    Subjects are numbered **sequentially** (0, 1, 2, ...) in ``subject_col``;
    any ``"patient_id"`` on the records is ignored, since synthetic patients do
    not correspond to real ones.

    Args:
        records: Iterable of ``{"visits": [[code, ...], ...]}`` dicts. Both the
            :class:`EHRGeneration` task output and a generator's ``generate()``
            output have this shape.
        label_fn: Optional callable mapping a record to a binary patient label
            (0/1) used by the utility metrics. Defaults to all-zeros.
        subject_col: Output patient-id column. Default ``"id"``.
        visit_col: Output visit-index column. Default ``"time"``.
        code_col: Output single-code column. Default ``"visit_codes"``.
        label_col: Output binary-label column. Default ``"labels"``.

    Returns:
        ``pandas.DataFrame`` with columns
        ``[subject_col, visit_col, code_col, label_col]``.

    Examples:
        >>> from pyhealth.tasks.generate_ehr import to_evaluation_dataframe
        >>> records = [{"visits": [["4019", "25000"], ["4019"]]}]
        >>> to_evaluation_dataframe(records)
           id  time visit_codes  labels
        0   0     0        4019       0
        1   0     0       25000       0
        2   0     1        4019       0
    """
    import pandas as pd

    rows = []
    for subject_id, record in enumerate(records):
        label = 0 if label_fn is None else int(label_fn(record))
        for visit_idx, visit in enumerate(record["visits"]):
            for code in visit:
                rows.append(
                    {
                        subject_col: subject_id,
                        visit_col: visit_idx,
                        code_col: code,
                        label_col: label,
                    }
                )
    return pd.DataFrame(
        rows, columns=[subject_col, visit_col, code_col, label_col]
    )


def decode_dataset(sample_dataset, feature_key: str = "visits") -> list[dict]:
    """Decode a processed EHRGeneration ``SampleDataset`` back into records.

    Inverts the :class:`~pyhealth.processors.NestedMultiHotProcessor` encoding
    using its vocabulary (skipping ``<pad>``/``<unk>``), yielding one
    ``{"visits": [[code_str, ...], ...]}`` record per sample. Use this to build
    the real train/test frames that ``evaluate_synthetic_ehr`` compares against.

    Codes come back in vocabulary order, not the order they were charted in,
    and repeats collapse -- the multi-hot form records presence, not sequence
    or count within a visit.

    Args:
        sample_dataset: A ``SampleDataset`` produced by :class:`EHRGeneration`.
        feature_key: Input feature key holding the nested code sequence.
            Default ``"visits"``.

    Returns:
        List of ``{"visits": [[code_str, ...], ...]}`` records.

    Raises:
        TypeError: If ``feature_key`` is not backed by a
            :class:`~pyhealth.processors.NestedMultiHotProcessor`.

    Examples:
        >>> from pyhealth.tasks.generate_ehr import decode_dataset
        >>> records = decode_dataset(samples)
        >>> records[0]["visits"][0]
        ['4019', '25000']
    """
    # split_by_patient hands back a torch Subset, which carries no processors
    # of its own -- decoding a split is the common case, so resolve through it.
    source = getattr(sample_dataset, "dataset", sample_dataset)
    processor = source.input_processors[feature_key]
    if not isinstance(processor, NestedMultiHotProcessor):
        raise TypeError(
            f"decode_dataset inverts the multi-hot encoding, but '{feature_key}' "
            f"is a {type(processor).__name__}. Use VisitMultiHotGeneration, or "
            "read the codes off the index tensor directly."
        )
    index_to_code = {idx: code for code, idx in processor.code_vocab.items()}

    records: list[dict] = []
    for i in range(len(sample_dataset)):
        sample = sample_dataset[i]
        visits: list[list[str]] = []
        # Each row is a multi-hot vector over the vocabulary, so the codes
        # present are its nonzero columns -- the values are all 1.0.
        for row in sample[feature_key]:
            codes = [
                index_to_code[int(col)]
                for col in row.nonzero(as_tuple=True)[0].tolist()
                if index_to_code.get(int(col)) not in (None, "<pad>", "<unk>")
            ]
            if codes:
                visits.append(codes)
        records.append({"visits": visits})
    return records
