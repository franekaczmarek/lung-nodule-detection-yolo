"""Balance the training split with rotated and zoomed copies of nodule slices.

Most CT slices contain no nodule. For every training patient this step creates
as many augmented nodule slices as needed to match the number of empty slices.
Each round rotates every usable nodule slice by a larger angle and zooms in
further around the nodules.

Usage:
    python -m lung_nodule_detection.augment \\
        --images data/nifti/images --labels data/nifti/labels --dataset data/yolo
"""

import argparse
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label as connected_components

from . import ct
from .config import MIN_AUGMENT_NODULE_AREA_MM2
from .dataset import (
    images_dir,
    label_is_empty,
    label_volume_name,
    labels_dir,
    read_dataset_info,
    reset_split,
    sample_stem,
    save_sample,
)
from .patients import patient_id

log = logging.getLogger(__name__)

AUGMENTED_SPLIT = "augmented"


@dataclass(frozen=True)
class Schedule:
    """Rotation and zoom used in each augmentation round."""

    rotation_step_deg: float = 19.0
    initial_zoom: float = 1.25
    zoom_step: float = 0.15
    max_zoom: float = 2.0
    max_rounds: int = 50

    def round(self, number: int) -> tuple[float, float]:
        """Rotation angle and zoom factor of round ``number`` (counted from 0)."""
        angle = self.rotation_step_deg * (number + 1)
        zoom = min(self.initial_zoom + self.zoom_step * number, self.max_zoom)
        return angle, zoom


def rotate(image: np.ndarray, mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an image and its mask about the centre, enlarging the frame so nothing is cut off."""
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w // 2
    matrix[1, 2] += new_h / 2 - h // 2
    image = cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST)
    return image, mask


def keep_largest_components(mask: np.ndarray, count: int) -> np.ndarray:
    """Keep only the ``count`` largest connected regions of a mask."""
    components, found = connected_components(mask > 0)
    if found <= count:
        return mask
    sizes = np.bincount(components.ravel())[1:]
    keep = np.argsort(-sizes)[:count] + 1
    return np.where(np.isin(components, keep), mask, 0).astype(mask.dtype)


def zoom_on_nodules(image: np.ndarray, mask: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    """Crop a window 1/zoom of the image size, centred on the nodule pixels."""
    ys, xs = np.nonzero(mask)
    center_y, center_x = int(ys.mean()), int(xs.mean())
    half_w, half_h = int(image.shape[1] / zoom) // 2, int(image.shape[0] / zoom) // 2
    top, left = max(0, center_y - half_h), max(0, center_x - half_w)
    bottom = min(image.shape[0], center_y + half_h)
    right = min(image.shape[1], center_x + half_w)
    return image[top:bottom, left:right], mask[top:bottom, left:right]


def augment_slice(
    hu: np.ndarray,
    nodule_mask: np.ndarray,
    lung_mask: np.ndarray,
    angle: float,
    zoom: float,
    canvas_size: tuple[int, int],
):
    """Crop to the lungs, rotate, zoom on the nodules and fit to the canvas.

    Returns ``(image, mask, content_bounds)``, or None when no nodule survives the rotation.
    """
    crop = ct.lung_bounding_box(lung_mask)
    image = ct.window_to_uint8(hu[crop])
    mask = (nodule_mask[crop] > 0).astype(np.uint8)
    _, original_regions = connected_components(mask)

    image, mask = rotate(image, mask, angle)
    # Nearest-neighbour rotation can split a nodule into slivers; drop the extra pieces.
    mask = keep_largest_components(mask, original_regions)
    if not mask.any():
        return None

    image, mask = zoom_on_nodules(image, mask, zoom)
    return ct.fit_to_canvas(image, mask, canvas_size, cv2.INTER_LINEAR)


def count_labels(dataset: Path, patients: set[str]) -> dict[str, Counter]:
    """Count empty and non-empty training label files per patient."""
    counts = {pid: Counter() for pid in patients}
    for label_file in labels_dir(dataset, "train").glob("*.txt"):
        pid = patient_id(label_file.name)
        if pid in counts:
            counts[pid]["empty" if label_is_empty(label_file) else "non_empty"] += 1
    return counts


def nodule_slices(image_path: Path, label_path: Path, segmenter, min_area_mm2: float) -> list[tuple]:
    """(slice index, HU slice, nodule mask, lung mask) for slices usable as augmentation sources."""
    volume = ct.read_volume(image_path)
    hu = ct.to_array(volume)
    nodule_masks = ct.to_array(ct.read_volume(label_path))
    spacing = ct.pixel_spacing_mm(volume)
    lung_masks = ct.segment_lungs(segmenter, volume)
    return [
        (index, hu[index], nodule_masks[index], lung_masks[index])
        for index in range(min(len(hu), len(nodule_masks)))
        if lung_masks[index].any() and ct.nodule_area_mm2(nodule_masks[index] > 0, spacing) >= min_area_mm2
    ]


def augment_patient(
    volume_name: str,
    sources: list[tuple],
    needed: int,
    dataset: Path,
    canvas_size: tuple[int, int],
    schedule: Schedule,
    start_index: int = 0,
) -> int:
    """Write up to ``needed`` augmented slices from ``sources``. Returns how many were written."""
    written = 0
    for round_number in range(schedule.max_rounds):
        angle, zoom = schedule.round(round_number)
        log.info("  round %d: rotation %.0f°, zoom %.2f", round_number + 1, angle, zoom)
        for index, hu, nodule_mask, lung_mask in sources:
            result = augment_slice(hu, nodule_mask, lung_mask, angle, zoom, canvas_size)
            if result is None:
                continue
            image, mask, bounds = result
            stem = sample_stem(volume_name, index, f"_augmented_{start_index + written}")
            save_sample(dataset, AUGMENTED_SPLIT, stem, image, mask, bounds)
            written += 1
            if written == needed:
                return written
    log.warning("  stopped after %d rounds with %d of %d images", schedule.max_rounds, written, needed)
    return written


def move_into_train(dataset: Path) -> None:
    """Move the augmented images and labels into the training split."""
    for source, target in (
        (images_dir(dataset, AUGMENTED_SPLIT), images_dir(dataset, "train")),
        (labels_dir(dataset, AUGMENTED_SPLIT), labels_dir(dataset, "train")),
    ):
        for path in source.iterdir():
            shutil.move(path, target / path.name)
    shutil.rmtree(dataset / AUGMENTED_SPLIT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, required=True, help="folder of CT volumes (*.nii.gz)")
    parser.add_argument("--labels", type=Path, required=True, help="folder of nodule mask volumes")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="dataset folder from the preprocessing step"
    )
    parser.add_argument("--lungmask-weights", type=Path, help="local R231 weights (downloaded if omitted)")
    parser.add_argument(
        "--min-nodule-area",
        type=float,
        default=MIN_AUGMENT_NODULE_AREA_MM2,
        help="only augment slices whose nodules cover at least this many mm²",
    )
    parser.add_argument(
        "--keep-separate",
        action="store_true",
        help=f"leave the results in {AUGMENTED_SPLIT}/ instead of moving them into train/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    info = read_dataset_info(args.dataset)
    canvas_size = tuple(info["image_size"])
    train_patients = set(info["patients"]["train"])
    counts = count_labels(args.dataset, train_patients)
    for pid in sorted(counts, key=int):
        log.info(
            "Patient %s: %d empty, %d non-empty labels", pid, counts[pid]["empty"], counts[pid]["non_empty"]
        )

    reset_split(args.dataset, AUGMENTED_SPLIT)
    segmenter = ct.make_lung_segmenter(args.lungmask_weights)
    schedule = Schedule()
    written_per_patient: Counter = Counter()

    for image_path in sorted(args.images.glob("*.nii.gz")):
        pid = patient_id(image_path.name)
        if pid not in train_patients:
            continue
        needed = counts[pid]["empty"] - counts[pid]["non_empty"] - written_per_patient[pid]
        if needed <= 0:
            log.info("Patient %s: already balanced", pid)
            continue
        label_path = args.labels / label_volume_name(image_path.name)
        if not label_path.exists():
            log.warning("Patient %s: missing %s, skipped", pid, label_path.name)
            continue

        log.info("Patient %s: generating %d images from %s", pid, needed, image_path.name)
        sources = nodule_slices(image_path, label_path, segmenter, args.min_nodule_area)
        if not sources:
            log.info("  no slices with nodules large enough to augment")
            continue
        written_per_patient[pid] += augment_patient(
            image_path.name,
            sources,
            needed,
            args.dataset,
            canvas_size,
            schedule,
            start_index=written_per_patient[pid],
        )

    log.info("Generated %d augmented images", sum(written_per_patient.values()))
    if not args.keep_separate:
        move_into_train(args.dataset)
        log.info("Moved them into %s", images_dir(args.dataset, "train").parent)


if __name__ == "__main__":
    main()
