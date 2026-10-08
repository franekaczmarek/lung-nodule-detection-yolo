"""End-to-end run of preprocessing and augmentation on synthetic CT volumes."""

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from lung_nodule_detection import augment, ct, preprocess  # noqa: E402
from lung_nodule_detection.dataset import (  # noqa: E402
    images_dir,
    label_is_empty,
    labels_dir,
    read_dataset_info,
)

SLICES, SIZE = 6, 64


class FakeLungSegmenter:
    """Marks a left and a right lung on every slice except the first."""

    def apply(self, volume):
        labels = np.zeros(sitk.GetArrayFromImage(volume).shape, dtype=np.uint8)
        labels[1:, 10:50, 8:28] = 1
        labels[1:, 10:50, 36:56] = 2
        return labels


def write_patient(folder, number):
    hu = np.full((SLICES, SIZE, SIZE), -800, dtype=np.int16)
    nodules = np.zeros_like(hu, dtype=np.uint8)
    nodules[3:5, 20:30, 12:22] = 1  # 10 x 10 px = 100 mm² at 1 mm spacing
    hu[nodules > 0] = 40
    for array, kind in ((hu, "Images"), (nodules, "Labels")):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((1.0, 1.0, 2.0))
        sitk.WriteImage(image, str(folder / kind.lower() / f"LIDC_IDRI_{number:04d}_{kind}.nii.gz"))


@pytest.fixture
def raw_data(tmp_path, monkeypatch):
    for kind in ("images", "labels"):
        (tmp_path / kind).mkdir()
    for number in range(1, 11):
        write_patient(tmp_path, number)
    monkeypatch.setattr(ct, "make_lung_segmenter", lambda weights=None: FakeLungSegmenter())
    return tmp_path


def test_preprocess_then_augment(raw_data):
    dataset = raw_data / "yolo"
    common = ["--images", str(raw_data / "images"), "--labels", str(raw_data / "labels")]

    preprocess.main([*common, "--output", str(dataset), "--seed", "0"])

    info = read_dataset_info(dataset)
    assert info["image_size"] == [48, 40]
    assert {split: len(p) for split, p in info["patients"].items()} == {"train": 7, "val": 2, "test": 1}
    train_labels = list(labels_dir(dataset, "train").glob("*.txt"))
    # 5 slices with lungs per patient, 2 of them with a nodule.
    assert len(train_labels) == 7 * 5
    assert sum(not label_is_empty(f) for f in train_labels) == 7 * 2
    assert (dataset / "dataset.yaml").read_text().startswith("path: ")

    augment.main([*common, "--dataset", str(dataset)])

    train_labels = list(labels_dir(dataset, "train").glob("*.txt"))
    assert len(train_labels) == 7 * 6
    assert sum(not label_is_empty(f) for f in train_labels) == 7 * 3
    assert len(list(images_dir(dataset, "train").glob("*.png"))) == len(train_labels)
    assert not (dataset / augment.AUGMENTED_SPLIT).exists()
