"""Nodule bounding boxes: extraction from masks, merging and YOLO labels.

A box is ``(x_min, x_max, y_min, y_max)`` in pixel coordinates, inclusive.
"""

import numpy as np
from scipy.ndimage import label as connected_components

from .config import BOX_MERGE_DISTANCE_PX, BOX_PADDING_PX, BOX_SCALE

Box = tuple[int, int, int, int]


def find_nodule_boxes(
    mask: np.ndarray,
    content_bounds: Box | None = None,
    scale: float = BOX_SCALE,
    padding: int = BOX_PADDING_PX,
) -> list[Box]:
    """Return one box per connected nodule region in a 2D mask.

    Each box is scaled around its centre, padded, clipped to the image and,
    when ``content_bounds`` is given, to the non-padded part of the image.
    Boxes that become empty after clipping are dropped.
    """
    components, count = connected_components(mask > 0)
    height, width = mask.shape
    boxes = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(components == index)
        x_center, y_center = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half_w = (xs.max() - xs.min()) * scale / 2 + padding
        half_h = (ys.max() - ys.min()) * scale / 2 + padding

        x_min = max(int(np.round(x_center - half_w)), 0)
        x_max = min(int(np.round(x_center + half_w)), width - 1)
        y_min = max(int(np.round(y_center - half_h)), 0)
        y_max = min(int(np.round(y_center + half_h)), height - 1)

        if content_bounds is not None:
            bx_min, bx_max, by_min, by_max = content_bounds
            x_min, x_max = max(x_min, bx_min), min(x_max, bx_max)
            y_min, y_max = max(y_min, by_min), min(y_max, by_max)
            if x_min >= x_max or y_min >= y_max:
                continue

        boxes.append((x_min, x_max, y_min, y_max))
    return boxes


def boxes_are_close(a: Box, b: Box, max_distance: int) -> bool:
    """True if two boxes overlap or are at most ``max_distance`` pixels apart."""
    ax_min, ax_max, ay_min, ay_max = a
    bx_min, bx_max, by_min, by_max = b
    x_close = ax_max + max_distance >= bx_min and bx_max + max_distance >= ax_min
    y_close = ay_max + max_distance >= by_min and by_max + max_distance >= ay_min
    return x_close and y_close


def union_box(a: Box, b: Box) -> Box:
    """The smallest box containing both boxes."""
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def merge_close_boxes(boxes: list[Box], max_distance: int = BOX_MERGE_DISTANCE_PX) -> list[Box]:
    """Merge each box into the first already-kept box it is close to."""
    merged: list[Box] = []
    for box in boxes:
        for i, kept in enumerate(merged):
            if boxes_are_close(box, kept, max_distance):
                merged[i] = union_box(box, kept)
                break
        else:
            merged.append(box)
    return merged


def to_yolo_lines(boxes: list[Box], image_shape: tuple[int, int], class_id: int) -> list[str]:
    """Format boxes as YOLO label lines: ``class x_center y_center width height``, normalised."""
    height, width = image_shape
    lines = []
    for x_min, x_max, y_min, y_max in boxes:
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        box_w = (x_max - x_min) / width
        box_h = (y_max - y_min) / height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    return lines
