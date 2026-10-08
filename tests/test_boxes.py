import numpy as np

from lung_nodule_detection.boxes import (
    boxes_are_close,
    find_nodule_boxes,
    merge_close_boxes,
    to_yolo_lines,
)


def test_box_is_scaled_and_padded_around_the_nodule():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:51, 30:41] = 1  # 10 px wide between first and last pixel

    # Half size: 10 * 1.3 / 2 + 5 = 11.5 around centre (35, 45).
    assert find_nodule_boxes(mask) == [(24, 46, 34, 56)]


def test_box_is_clipped_to_content_and_dropped_when_empty():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:51, 30:41] = 1

    assert find_nodule_boxes(mask, content_bounds=(30, 99, 0, 99)) == [(30, 46, 34, 56)]
    assert find_nodule_boxes(mask, content_bounds=(60, 99, 0, 99)) == []


def test_one_box_per_connected_region():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:15, 10:15] = 1
    mask[70:75, 70:75] = 1

    assert len(find_nodule_boxes(mask)) == 2


def test_close_boxes_are_merged_and_far_boxes_kept():
    assert boxes_are_close((0, 10, 0, 10), (20, 30, 0, 10), max_distance=10)
    assert not boxes_are_close((0, 10, 0, 10), (21, 30, 0, 10), max_distance=10)

    merged = merge_close_boxes([(0, 10, 0, 10), (15, 20, 5, 12), (80, 90, 80, 90)])
    assert merged == [(0, 20, 0, 12), (80, 90, 80, 90)]


def test_yolo_lines_are_normalised():
    assert to_yolo_lines([(0, 50, 0, 25)], (100, 200), class_id=0) == [
        "0 0.125000 0.125000 0.250000 0.250000"
    ]
