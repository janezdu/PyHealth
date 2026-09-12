pyhealth.tasks.generate_ehr
===========================================

Tasks that turn a longitudinal EHR dataset into training samples for
unconditional synthetic-EHR generators, plus helpers to flatten generated
output into the long-form dataframe consumed by
:mod:`pyhealth.metrics.generative`.

Extraction is shared; the encoding is not. Each generator family reads its
codes in a different shape, and handing a model the wrong shape fails silently
rather than loudly, so pick the task that matches the model:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Task
     - Encoding
     - Models
   * - ``VisitMultiHotGeneration``
     - one multi-hot row per visit
     - HALO
   * - ``VisitSequenceGeneration``
     - per-visit code indices
     - GPT2, PromptEHR
   * - ``PatientCodeSetGeneration``
     - one code set per patient
     - MedGAN, CorGAN

Task Classes
------------

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGeneration
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.VisitMultiHotGeneration
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.VisitSequenceGeneration
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.PatientCodeSetGeneration
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGenerationMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.generate_ehr.EHRGenerationMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

Helper Functions
----------------

.. autofunction:: pyhealth.tasks.generate_ehr.decode_dataset

.. autofunction:: pyhealth.tasks.generate_ehr.to_evaluation_dataframe
