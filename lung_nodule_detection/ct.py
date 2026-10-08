"""Reading CT volumes, lung segmentation and slice-level image operations."""

from pathlib import Path

import cv2
import numpy as np

from .boxes import Box
from .config import LUNGMASK_MODEL, WINDOW_CENTER_HU, WINDOW_WIDTH_HU


def read_volume(path: Path):
    """Read a NIfTI volume as a SimpleITK image."""
    import SimpleITK as sitk

    return sitk.ReadImage(str(path))


def to_array(volume) -> np.ndarray:
    """Voxel array of a SimpleITK image, shaped (slices, rows, columns)."""
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(volume)


def pixel_spacing_mm(volume) -> tuple[float, float]:
    """In-plane pixel spacing (x, y) of a SimpleITK image, in millimetres."""
    spacing = volume.GetSpacing()
    return spacing[0], spacing[1]


def make_lung_segmenter(weights_path: Path | None = None):
    """Create the lungmask R231 segmenter. Without weights, lungmask downloads them."""
    from lungmask import LMInferer

    if weights_path is None:
        return LMInferer(modelname=LUNGMASK_MODEL)
    return LMInferer(modelname=LUNGMASK_MODEL, modelpath=str(weights_path))


def segment_lungs(segmenter, volume) -> np.ndarray:
    """Boolean lung mask for every slice; lungmask labels the right and left lung 1 and 2."""
    labels = segmenter.apply(volume)
    return np.isin(labels, (1, 2))


def lung_bounding_box(lung_mask: np.ndarray) -> tuple[slice, slice] | None:
    """Row and column slices that crop a 2D mask to its lungs, or None without lungs."""
    rows = np.flatnonzero(lung_mask.any(axis=1))
    cols = np.flatnonzero(lung_mask.any(axis=0))
    if rows.size == 0:
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def has_both_lungs(lung_mask: np.ndarray) -> bool:
    """True if both the left and right half of a 2D slice contain lung tissue."""
    middle = lung_mask.shape[1] // 2
    return bool(lung_mask[:, :middle].any() and lung_mask[:, middle:].any())


def window_to_uint8(
    hu: np.ndarray, center: float = WINDOW_CENTER_HU, width: float = WINDOW_WIDTH_HU
) -> np.ndarray:
    """Clip Hounsfield units to a window and scale them to 0-255."""
    low, high = center - width / 2, center + width / 2
    scaled = (np.clip(hu, low, high) - low) / (high - low) * 255
    return scaled.astype(np.uint8)


def nodule_area_mm2(mask: np.ndarray, spacing: tuple[float, float]) -> float:
    """Area covered by nodule pixels of a 2D mask at its original pixel spacing."""
    return float(np.count_nonzero(mask) * spacing[0] * spacing[1])


def fit_to_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray, Box]:
    """Resize an image and its mask to fit ``size`` (width, height), keeping the aspect ratio.

    The rest of the canvas is zero-padded evenly on both sides. Returns the image,
    the mask and the box ``(x_min, x_max, y_min, y_max)`` covered by real content.
    """
    canvas_w, canvas_h = size
    scale = min(canvas_w / image.shape[1], canvas_h / image.shape[0])
    new_w, new_h = int(image.shape[1] * scale), int(image.shape[0] * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    left, top = (canvas_w - new_w) // 2, (canvas_h - new_h) // 2
    right, bottom = canvas_w - new_w - left, canvas_h - new_h - top
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return image, mask, (left, left + new_w - 1, top, top + new_h - 1)
