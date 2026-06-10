"""Task function for HALO synthetic data generation."""

from typing import Any, Dict, List

import polars as pl

from pyhealth.tasks.base_task import BaseTask


class HaloGenerationMIMIC3(BaseTask):
    """Task for HALO synthetic data generation using MIMIC-III dataset.

    HALO trains an autoregressive transformer to generate synthetic EHR data.
    This task extracts diagnosis code sequences per patient, where each patient
    produces one sample containing all their visits.

    Each sample contains all admissions for a patient, with the ICD-9 diagnosis
    codes for each admission grouped into a nested list. Patients with fewer than
    2 admissions that contain diagnosis codes are excluded.

    Attributes:
        task_name (str): "HaloGenerationMIMIC3"
        input_schema (Dict[str, str]): {"visits": "nested_sequence"}
        output_schema (Dict[str, str]): {} (generative task, no prediction target)
        _icd_col (str): Polars column name for ICD codes in diagnoses_icd table.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import HaloGenerationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="path/to/mimic3",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = HaloGenerationMIMIC3()
        >>> sample_dataset = dataset.set_task(task)
        >>> sample_dataset[0]  # doctest: +ELLIPSIS
        {'patient_id': ..., 'visits': [[...], ...]}
    """

    task_name: str = "HaloGenerationMIMIC3"
    input_schema: Dict[str, str] = {"visits": "nested_sequence"}
    output_schema: Dict[str, str] = {}
    _icd_col: str = "diagnoses_icd/icd9_code"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient for HALO generation task.

        Extracts diagnosis codes per admission, grouped by visit. Returns one
        sample per patient containing all visits with diagnosis codes. Excludes
        patients with fewer than 2 visits containing diagnosis codes.

        Args:
            patient: Patient object (PyHealth 2.0 Polars-based API)

        Returns:
            List of at most one sample dict:
                {
                    "patient_id": str,
                    "visits": [[code1, code2, ...], [code3, ...], ...]
                }
            Returns empty list if patient has fewer than 2 admissions with
            diagnosis codes.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) < 2:
            return []

        visits = []
        for admission in admissions:
            diagnoses_df = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            codes = (
                diagnoses_df.select(pl.col(self._icd_col))
                .to_series()
                .drop_nulls()
                .to_list()
            )
            if len(codes) > 0:
                visits.append(codes)

        if len(visits) < 2:
            return []

        return [{"patient_id": patient.patient_id, "visits": visits}]


class HaloGenerationMIMIC4(HaloGenerationMIMIC3):
    """Task for HALO synthetic data generation using MIMIC-IV dataset.

    Same logic as HaloGenerationMIMIC3. MIMIC-IV stores ICD codes under
    ``diagnoses_icd/icd_code`` rather than ``diagnoses_icd/icd9_code``.

    Attributes:
        task_name (str): "HaloGenerationMIMIC4"
        _icd_col (str): "diagnoses_icd/icd_code"
    """

    task_name: str = "HaloGenerationMIMIC4"
    _icd_col: str = "diagnoses_icd/icd_code"


# Convenience callable instances (same pattern as other PyHealth tasks)
halo_generation_mimic3_fn = HaloGenerationMIMIC3()
halo_generation_mimic4_fn = HaloGenerationMIMIC4()
