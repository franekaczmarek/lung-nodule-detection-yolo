"""Turn LIDC-IDRI CT volumes and nodule masks into a YOLO dataset of axial slices.

For every slice with both lungs visible, the slice is cropped to the lungs,
windowed to 8-bit, resized onto a canvas the size of the largest lung crop and
saved with a YOLO label file (empty for slices without nodules). Patients are
split into train, val and test before any slice is written.

Usage:
    python -m lung_nodule_detection.preprocess \\
        --images data/nifti/images --labels data/nifti/labels --output data/yolo
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import cv2

from . import ct
from .config import MIN_TRAIN_NODULE_AREA_MM2, SPLIT_RATIOS
from .dataset import (
    label_volume_name,
    reset_split,
    sample_stem,
    save_sample,
    write_dataset_info,
)
from .patients import split_by_patient

log = logging.getLogger(__name__)


def largest_lung_crop(volume_paths: list[Path], segmenter) -> tuple[int, int]:
    """Width and height of the largest lung bounding box over all slices of all volumes."""
    max_w = max_h = 0
    for path in volume_paths:
        log.info("Measuring lungs in %s", path.name)
        lung_masks = ct.segment_lungs(segmenter, ct.read_volume(path))
        for lung_mask in lung_masks:
            crop = ct.lung_bounding_box(lung_mask)
            if crop is not None:
                rows, cols = crop
                max_h = max(max_h, rows.stop - rows.start)
                max_w = max(max_w, cols.stop - cols.start)
    return int(max_w), int(max_h)


def process_volume(
    image_path: Path,
    label_path: Path,
    segmenter,
    output: Path,
    split: str,
    canvas_size: tuple[int, int],
    min_nodule_area_mm2: float,
) -> Counter:
    """Write every usable slice of one CT volume to ``split``. Returns slice counts."""
    stats: Counter = Counter()
    volume = ct.read_volume(image_path)
    hu = ct.to_array(volume)
    nodule_masks = ct.to_array(ct.read_volume(label_path))
    if hu.shape[0] != nodule_masks.shape[0]:
        log.warning(
            "Skipping %s: %d image slices but %d label slices",
            image_path.name,
            hu.shape[0],
            nodule_masks.shape[0],
        )
        stats["skipped_volumes"] += 1
        return stats

    spacing = ct.pixel_spacing_mm(volume)
    lung_masks = ct.segment_lungs(segmenter, volume)

    for index, (hu_slice, nodule_mask, lung_mask) in enumerate(
        zip(hu, nodule_masks, lung_masks, strict=True)
    ):
        crop = ct.lung_bounding_box(lung_mask)
        if crop is None:
            stats["no_lungs"] += 1
            continue
        if not ct.has_both_lungs(lung_mask):
            stats["one_lung"] += 1
            continue

        nodule_crop = nodule_mask[crop] > 0
        has_nodule = nodule_crop.any()
        if has_nodule:
            stats["with_nodules"] += 1
            # Tiny nodules are hard to label reliably, so they are left out of training.
            if split == "train" and ct.nodule_area_mm2(nodule_crop, spacing) < min_nodule_area_mm2:
                stats["small_nodules_removed"] += 1
                continue

        image, mask, bounds = ct.fit_to_canvas(
            ct.window_to_uint8(hu_slice[crop]), nodule_crop, canvas_size, cv2.INTER_LANCZOS4
        )
        stem = sample_stem(image_path.name, index)
        if has_nodule and mask.any():
            save_sample(output, split, stem, image, mask, bounds)
            stats["saved_with_nodules"] += 1
        else:
            save_sample(output, split, stem, image)
        stats["saved"] += 1
    return stats


def log_patient_stats(split: str, pid: str, stats: Counter) -> None:
    log.info(
        "[%s] patient %s: saved %d slices (%d with nodules); skipped %d without lungs, "
        "%d with one lung, %d with nodules under the area threshold",
        split,
        pid,
        stats["saved"],
        stats["saved_with_nodules"],
        stats["no_lungs"],
        stats["one_lung"],
        stats["small_nodules_removed"],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, required=True, help="folder of CT volumes (*.nii.gz)")
    parser.add_argument("--labels", type=Path, required=True, help="folder of nodule mask volumes")
    parser.add_argument("--output", type=Path, required=True, help="dataset folder to create")
    parser.add_argument("--lungmask-weights", type=Path, help="local R231 weights (downloaded if omitted)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the patient split")
    parser.add_argument(
        "--split", type=float, nargs=3, default=SPLIT_RATIOS, metavar=("TRAIN", "VAL", "TEST")
    )
    parser.add_argument(
        "--min-nodule-area",
        type=float,
        default=MIN_TRAIN_NODULE_AREA_MM2,
        help="drop training slices whose nodules cover fewer mm²",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    volume_names = sorted(p.name for p in args.images.glob("*.nii.gz"))
    if not volume_names:
        raise SystemExit(f"No .nii.gz volumes found in {args.images}")
    splits = split_by_patient(volume_names, tuple(args.split), args.seed)
    for split, patients in splits.items():
        log.info("%s: %d patients, %d volumes", split, len(patients), sum(map(len, patients.values())))

    segmenter = ct.make_lung_segmenter(args.lungmask_weights)
    canvas_size = largest_lung_crop([args.images / name for name in volume_names], segmenter)
    log.info("Canvas size (largest lung crop): %d x %d", *canvas_size)

    for split, patients in splits.items():
        reset_split(args.output, split)
        for pid, names in patients.items():
            stats: Counter = Counter()
            for name in names:
                log.info("Processing %s", name)
                stats += process_volume(
                    args.images / name,
                    args.labels / label_volume_name(name),
                    segmenter,
                    args.output,
                    split,
                    canvas_size,
                    args.min_nodule_area,
                )
            log_patient_stats(split, pid, stats)

    write_dataset_info(
        args.output, canvas_size, {split: list(patients) for split, patients in splits.items()}
    )
    log.info("Dataset written to %s", args.output)


if __name__ == "__main__":
    main()
