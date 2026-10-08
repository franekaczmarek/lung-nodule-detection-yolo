import numpy as np

from lung_nodule_detection import ct


def test_window_maps_window_edges_to_0_and_255():
    hu = np.array([-2000, -850, -350, 150, 3000])
    assert ct.window_to_uint8(hu).tolist() == [0, 0, 127, 255, 255]


def test_lung_bounding_box_and_both_lungs():
    mask = np.zeros((10, 20), dtype=bool)
    assert ct.lung_bounding_box(mask) is None

    mask[2:6, 3:5] = True
    assert ct.lung_bounding_box(mask) == (slice(2, 6), slice(3, 5))
    assert not ct.has_both_lungs(mask)

    mask[2:6, 14:17] = True
    assert ct.has_both_lungs(mask)


def test_nodule_area_uses_pixel_spacing():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:2, :5] = True
    assert ct.nodule_area_mm2(mask, (0.5, 0.8)) == 4.0


def test_fit_to_canvas_keeps_aspect_ratio_and_centres_content():
    image = np.full((50, 100), 200, dtype=np.uint8)
    mask = np.ones((50, 100), dtype=np.uint8)

    out_image, out_mask, bounds = ct.fit_to_canvas(image, mask, (200, 160))

    assert out_image.shape == out_mask.shape == (160, 200)
    assert bounds == (0, 199, 30, 129)
    assert out_mask[30:130].all() and not out_mask[:30].any() and not out_mask[130:].any()
