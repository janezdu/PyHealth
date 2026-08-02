"""Unit tests for the rare-code stratified cohort splitter.

Pure numpy -- no eICU, no torch, no GPU. These run as a pre-flight step inside
``run_freeze_split.sh`` so a stratifier bug fails the job in seconds instead of
after the multi-hour eICU load.
"""

import os
import random
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "examples",
        "fedpyhealth",
    ),
)

from freeze_cohort_split import (  # noqa: E402
    compute_rare_codes,
    iterative_stratified_split,
    verify_split,
)


def _disjoint_cohort(sizes):
    """Build patients where rare code ``c{n}`` is carried by exactly n patients.

    Codes do not co-occur, so each code's realized split is exactly its quota --
    which is what makes the quota table directly observable.
    """
    patient_codes = {}
    for n in sizes:
        for i in range(n):
            patient_codes[f"p{n}_{i:03d}"] = [f"c{n}", "common"]
    return patient_codes


# --------------------------------------------------------------------------- #
# Quotas                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n,expected",
    [(2, (1, 1)), (3, (2, 1)), (4, (3, 1)), (5, (4, 1)),
     (7, (6, 1)), (8, (6, 2)), (9, (7, 2)), (10, (8, 2))],
)
def test_quota_table(n, expected):
    """round(0.2*n), floored at 1 and capped at n-1, on both sides."""
    patient_codes = _disjoint_cohort([n])
    rare = [f"c{n}"]
    train, val = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    per_code = {
        "n_train": sum(1 for p in train if f"c{n}" in patient_codes[p]),
        "n_val": sum(1 for p in val if f"c{n}" in patient_codes[p]),
    }
    assert (per_code["n_train"], per_code["n_val"]) == expected


def test_every_rare_code_lands_in_both_folds():
    """The guarantee that keeps rare prevalence from being a structural zero."""
    patient_codes = _disjoint_cohort(range(2, 11))
    rare = [f"c{n}" for n in range(2, 11)]
    train, val = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    for code in rare:
        assert any(code in patient_codes[p] for p in train), code
        assert any(code in patient_codes[p] for p in val), code


# --------------------------------------------------------------------------- #
# Partition invariants                                                         #
# --------------------------------------------------------------------------- #
def test_folds_are_disjoint_and_exhaustive():
    patient_codes = _disjoint_cohort(range(2, 11))
    rare = [f"c{n}" for n in range(2, 11)]
    train, val = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(patient_codes)


def test_patients_with_no_rare_code_are_still_assigned():
    patient_codes = _disjoint_cohort([5])
    for i in range(50):
        patient_codes[f"plain{i:03d}"] = ["common"]
    train, val = iterative_stratified_split(patient_codes, ["c5"], 0.2, seed=0)
    assert set(train) | set(val) == set(patient_codes)


# --------------------------------------------------------------------------- #
# Determinism -- the property the frozen manifest depends on                   #
# --------------------------------------------------------------------------- #
def test_determinism_same_seed():
    patient_codes = _disjoint_cohort(range(2, 9))
    rare = [f"c{n}" for n in range(2, 9)]
    a = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    b = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    assert a == b


def test_order_invariance():
    """Shuffling the input dict must not change the split.

    The splitter sorts its inputs precisely so a manifest cannot silently drift
    when the dataset build changes iteration order.
    """
    patient_codes = _disjoint_cohort(range(2, 9))
    rare = [f"c{n}" for n in range(2, 9)]
    baseline = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)

    items = list(patient_codes.items())
    random.Random(1234).shuffle(items)
    shuffled = dict(items)
    assert list(shuffled) != list(patient_codes)  # the shuffle did something
    assert iterative_stratified_split(shuffled, rare, 0.2, seed=0) == baseline


def test_rare_code_order_invariance():
    patient_codes = _disjoint_cohort(range(2, 9))
    rare = [f"c{n}" for n in range(2, 9)]
    baseline = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    assert iterative_stratified_split(
        patient_codes, list(reversed(rare)), 0.2, seed=0
    ) == baseline


# --------------------------------------------------------------------------- #
# Realized fraction                                                            #
# --------------------------------------------------------------------------- #
def test_realized_val_fraction_on_realistic_overlap():
    """With co-occurring codes the >=1-per-fold guarantee stays affordable."""
    rng = random.Random(0)
    codes = [f"r{i:03d}" for i in range(120)]
    patient_codes = {}
    for i in range(1500):
        picked = rng.sample(codes, rng.randint(1, 6))
        patient_codes[f"p{i:04d}"] = sorted(picked + ["common"])
    rare = compute_rare_codes(patient_codes, 0.05, 2)
    train, val = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    frac = len(val) / len(patient_codes)
    assert 0.15 <= frac <= 0.30, frac


def test_disjoint_two_patient_codes_inflate_val_fraction():
    """The known cost of the guarantee, pinned so it is a decision not a surprise.

    When rare codes never co-occur, every 2-patient code forces one of its two
    patients into validation, so the realized fraction runs far above 0.2. Real
    hospitals have heavy co-occurrence (previous test), but this is why
    ``verify_split`` reports the realized fraction rather than assuming 0.2.
    """
    patient_codes = {
        f"p{c:02d}_{i}": [f"c{c:02d}"] for c in range(40) for i in range(2)
    }
    rare = [f"c{c:02d}" for c in range(40)]
    train, val = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    assert len(val) == 40  # exactly one per code
    assert len(val) / len(patient_codes) == 0.5


# --------------------------------------------------------------------------- #
# Rare-code definition                                                         #
# --------------------------------------------------------------------------- #
def test_compute_rare_codes_thresholds():
    patient_codes = {f"p{i:03d}": [] for i in range(100)}
    for i in range(100):
        patient_codes[f"p{i:03d}"] = ["common"]          # 100% -> not rare
    for i in range(4):
        patient_codes[f"p{i:03d}"].append("rare4")       # 4% -> rare
    patient_codes["p000"].append("singleton")            # 1 patient -> excluded
    for i in range(6):
        patient_codes[f"p{i:03d}"].append("borderline")  # 6% -> not rare

    rare = compute_rare_codes(patient_codes, 0.05, 2)
    assert set(rare) == {"rare4"}
    assert rare["rare4"]["n_patients"] == 4
    assert rare["rare4"]["prevalence"] == pytest.approx(0.04)


def test_compute_rare_codes_raises_on_empty_set():
    """The --dev trap: 5% of 20 patients is < the 2-patient floor."""
    patient_codes = {f"p{i}": ["a", "b"] for i in range(20)}
    with pytest.raises(ValueError, match="empty rare set"):
        compute_rare_codes(patient_codes, 0.05, 2)


def test_compute_rare_codes_boundary_is_inclusive():
    patient_codes = {f"p{i:03d}": ["common"] for i in range(100)}
    for i in range(5):
        patient_codes[f"p{i:03d}"].append("exactly5pct")
    rare = compute_rare_codes(patient_codes, 0.05, 2)
    assert "exactly5pct" in rare


# --------------------------------------------------------------------------- #
# verify_split                                                                 #
# --------------------------------------------------------------------------- #
def test_verify_split_accepts_a_good_split():
    patient_codes = _disjoint_cohort(range(2, 11))
    rare = [f"c{n}" for n in range(2, 11)]
    train, val = iterative_stratified_split(patient_codes, rare, 0.2, seed=0)
    per_code = verify_split(patient_codes, rare, train, val, 0.2, tol=1.0)
    assert set(per_code) == set(rare)
    assert all(v["n_train"] >= 1 and v["n_val"] >= 1 for v in per_code.values())


def test_verify_split_rejects_a_missing_rare_code():
    patient_codes = _disjoint_cohort([4])
    train = [p for p in patient_codes]      # everything in train, val empty
    with pytest.raises(ValueError, match="guarantee was violated"):
        verify_split(patient_codes, ["c4"], train, [], 0.2, tol=1.0)


def test_verify_split_rejects_overlap():
    patient_codes = _disjoint_cohort([4])
    ids = sorted(patient_codes)
    with pytest.raises(ValueError, match="overlap"):
        verify_split(patient_codes, ["c4"], ids, ids[:1], 0.2, tol=1.0)


def test_verify_split_rejects_unassigned_patients():
    patient_codes = _disjoint_cohort([4])
    ids = sorted(patient_codes)
    with pytest.raises(ValueError, match="unassigned"):
        verify_split(patient_codes, ["c4"], ids[:2], ids[2:3], 0.2, tol=1.0)
