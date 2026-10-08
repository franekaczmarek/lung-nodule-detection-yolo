import pytest

from lung_nodule_detection.patients import patient_id, split_by_patient


def test_patient_id_strips_leading_zeros():
    assert patient_id("LIDC_IDRI_0042_Images.nii.gz") == "42"
    assert patient_id("scan.nii.gz") is None


def test_split_keeps_each_patient_in_one_split():
    files = [f"LIDC_IDRI_{p:04d}_{part}_Images.nii.gz" for p in range(1, 11) for part in "ab"]

    splits = split_by_patient(files, seed=0)

    assert {split: len(patients) for split, patients in splits.items()} == {"train": 7, "val": 2, "test": 1}
    seen = [pid for patients in splits.values() for pid in patients]
    assert sorted(seen, key=int) == [str(p) for p in range(1, 11)]
    assert all(len(names) == 2 for patients in splits.values() for names in patients.values())


def test_split_is_reproducible_with_a_seed():
    files = [f"LIDC_IDRI_{p:04d}_Images.nii.gz" for p in range(20)]
    assert split_by_patient(files, seed=1) == split_by_patient(files, seed=1)


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        split_by_patient(["LIDC_IDRI_0001.nii.gz"], ratios=(0.5, 0.5, 0.5))
