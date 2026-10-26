import numpy as np

HU_MIN = -200.0
HU_MAX = 300.0


def normalize_ct(ct):
    ct = np.asarray(ct, dtype=np.float32)
    ct = np.clip(ct, HU_MIN, HU_MAX)
    return ((ct - HU_MIN) / (HU_MAX - HU_MIN + 1e-8)).astype(np.float32)


def get_bbox(mask, margin=20):
    pos = np.where(np.asarray(mask) > 0)
    if len(pos[0]) == 0:
        raise ValueError("mask is empty")
    starts = [max(0, int(axis.min()) - margin) for axis in pos]
    stops = [min(mask.shape[i], int(axis.max()) + 1 + margin)
             for i, axis in enumerate(pos)]
    return starts[0], stops[0], starts[1], stops[1], starts[2], stops[2]
