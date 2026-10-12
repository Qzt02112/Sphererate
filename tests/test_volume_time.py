import numpy as np

from sphererate.baselines.volume_time import (
    temporal_volume_to_cumulative_maps,
    uniform_slice_overlap_weights,
)
from sphererate.data.maps import MapGeometry


def test_uniform_slice_overlap_weights() -> None:
    np.testing.assert_allclose(uniform_slice_overlap_weights(0.5, num_slices=12, duration_s=24.0)[:2], [0.5, 0.0])
    np.testing.assert_allclose(uniform_slice_overlap_weights(3.0, num_slices=12, duration_s=24.0)[:3], [2.0, 1.0, 0.0])


def test_temporal_volume_accumulation_is_spherically_normalized() -> None:
    volume = np.zeros((12, 8, 16), dtype=np.float32)
    volume[0, 4, 2] = 1.0
    volume[1, 4, 10] = 1.0
    maps = temporal_volume_to_cumulative_maps(
        volume,
        horizons_s=(0.5, 1.0, 3.0),
        duration_s=24.0,
        height=16,
        width=32,
    )
    area = MapGeometry(16, 32).area_weights
    np.testing.assert_allclose((maps * area).sum(axis=(1, 2)), 1.0, atol=1e-5)
    np.testing.assert_allclose(maps[0], maps[1], atol=0.0)
    assert not np.allclose(maps[1], maps[2])
