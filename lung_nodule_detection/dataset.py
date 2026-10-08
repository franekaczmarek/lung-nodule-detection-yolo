"""Layout of the YOLO dataset on disk and writing samples into it."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .boxes import Box, find_nodule_boxes, merge_close_boxes, to_yolo_lines
from .config import (
    DATASET_INFO_FILE,
    DATASET_YAML_FILE,
    NODULE_CLASS_ID,
    NODULE_CLASS_NAME,
    SPLITS,
)


def label_volume_name(image_volume_name: str) -> str:
    """Name of the nodule mask volume that belongs to a CT volume."""
    return image_volume_name.replace("Images", "Labels")


def volume_stem(volume_name: str) -> str:
    return volume_name.removesuffix(".nii.gz")


def sample_stem(volume_name: str, slice_index: int, suffix: str = "") -> str:
    return f"{volume_stem(volume_name)}_slice_{slice_index}_axis_z{suffix}"


def images_dir(root: Path, split: str) -> Path:
    return root / split / "images"


def labels_dir(root: Path, split: str) -> Path:
    return root / split / "labels"


def reset_split(root: Path, split: str) -> None:
    """Delete and recreate the image and label folders of a split."""
    for folder in (images_dir(root, split), labels_dir(root, split)):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)


def save_sample(
    root: Path,
    split: str,
    stem: str,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    content_bounds: Box | None = None,
) -> int:
    """Write a PNG slice and its YOLO label file. Returns the number of boxes written.

    Without a mask, or with an empty one, the label file is empty: a negative sample.
    """
    cv2.imwrite(str(images_dir(root, split) / f"{stem}.png"), image)

    lines: list[str] = []
    if mask is not None:
        boxes = merge_close_boxes(find_nodule_boxes(mask, content_bounds))
        lines = to_yolo_lines(boxes, image.shape[:2], NODULE_CLASS_ID)
    (labels_dir(root, split) / f"{stem}.txt").write_text("\n".join(lines))
    return len(lines)


def label_is_empty(label_file: Path) -> bool:
    return not label_file.read_text().strip()


def write_dataset_info(root: Path, image_size: tuple[int, int], patients: dict[str, list[str]]) -> None:
    """Record the canvas size and patient split, and write the Ultralytics dataset config."""
    info = {"image_size": list(image_size), "patients": patients}
    (root / DATASET_INFO_FILE).write_text(json.dumps(info, indent=2))

    yaml_lines = [f"path: {root.resolve().as_posix()}"]
    yaml_lines += [f"{split}: {split}/images" for split in SPLITS]
    yaml_lines += ["names:", f"  {NODULE_CLASS_ID}: {NODULE_CLASS_NAME}", ""]
    (root / DATASET_YAML_FILE).write_text("\n".join(yaml_lines))


def read_dataset_info(root: Path) -> dict:
    info_file = root / DATASET_INFO_FILE
    if not info_file.exists():
        raise FileNotFoundError(f"{info_file} not found. Run the preprocessing step first.")
    return json.loads(info_file.read_text())
