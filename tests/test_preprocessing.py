import numpy as np

from gist_net.preprocessing import get_bbox, normalize_ct


def test_normalize_ct_bounds():
    out = normalize_ct(np.array([-500.0, -200.0, 300.0, 800.0]))
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_bbox_contains_mask():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[8:12, 7:13, 9:11] = 1
    assert get_bbox(mask, margin=2) == (6, 14, 5, 15, 7, 13)
