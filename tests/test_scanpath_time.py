import numpy as np

from sphererate.baselines.scanpath_time import (
    official_second_step_weights,
    second_sampled_scanpaths_to_cumulative_maps,
)
from sphererate.data.maps import MapGeometry


def test_official_second_step_weights_marks_subsecond_as_first_second_approximation() -> None:
    assert official_second_step_weights(0.5, 6).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert official_second_step_weights(1.0, 6).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert official_second_step_weights(3.0, 6).tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert official_second_step_weights(5.0, 3).tolist() == [1.0, 1.0, 1.0]


def test_second_sampled_scanpaths_produce_spherical_densities() -> None:
    scanpaths = np.array(
        [
            [[0.5, 0.25], [0.5, 0.50], [0.5, 0.75]],
            [[0.5, 0.25], [0.5, 0.50], [0.5, 0.75]],
        ],
        dtype=np.float32,
    )
    maps = second_sampled_scanpaths_to_cumulative_maps(
        scanpaths,
        horizons_s=(0.5, 1.0, 3.0),
        height=32,
        width=64,
        sigma_px=2.0,
    )
    area = MapGeometry(32, 64).area_weights
    assert maps.shape == (3, 32, 64)
    np.testing.assert_allclose((maps * area).sum(axis=(1, 2)), np.ones(3), atol=1e-5)
    np.testing.assert_allclose(maps[0], maps[1], atol=0.0)
    assert not np.allclose(maps[1], maps[2])
