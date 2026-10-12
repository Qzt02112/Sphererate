from __future__ import annotations

import numpy as np

from sphererate.evaluation.rollout import peak_to_norm, prefix_maps_from_history


def test_peak_to_norm_uses_pixel_centers() -> None:
    pred = np.zeros((10, 20), dtype=np.float32)
    pred[4, 7] = 1.0
    x, y = peak_to_norm(pred)
    assert x == (7.5 / 20.0)
    assert y == (4.5 / 10.0)


def test_prefix_maps_left_pad_missing_history() -> None:
    area = np.ones((16, 32), dtype=np.float32)
    prefix = prefix_maps_from_history(
        [(0.25, 0.5)],
        prefix_length=3,
        height=16,
        width=32,
        area_weights=area,
        sigma_px=1.0,
        area_weighted=False,
    )
    assert prefix.shape == (3, 16, 32)
    assert float(prefix[0].sum()) == 0.0
    assert float(prefix[1].sum()) == 0.0
    assert float(prefix[2].sum()) > 0.0
