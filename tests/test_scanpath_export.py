from __future__ import annotations

import numpy as np

from sphererate.baselines.scanpath_export import (
    declared_start_xyz,
    erp_xy_to_xyz,
    sample_observer_seconds,
    xyz_to_erp_yx,
)


def test_erp_xyz_round_trip_preserves_yx() -> None:
    x = np.asarray([0.0, 0.25, 0.75, 0.999], dtype=np.float32)
    y = np.asarray([0.1, 0.4, 0.5, 0.9], dtype=np.float32)
    round_trip = xyz_to_erp_yx(erp_xy_to_xyz(x, y))
    np.testing.assert_allclose(round_trip[:, 0], y, atol=1e-6)
    circular_error = np.minimum(np.abs(round_trip[:, 1] - x), 1.0 - np.abs(round_trip[:, 1] - x))
    assert float(circular_error.max()) < 1e-6


def test_declared_start_is_pitch_zero_yaw_plus_90() -> None:
    np.testing.assert_allclose(declared_start_xyz(0.0, 90.0), [0.0, 1.0, 0.0], atol=1e-6)


def test_second_sampling_repairs_one_missing_bin() -> None:
    # x, y, start, observer, duration, unused temporal bin
    fixations = np.asarray(
        [
            [0.75, 0.50, 0.1, 3, 0.8, 0],
            [0.80, 0.55, 2.1, 3, 0.8, 0],
            [0.85, 0.60, 3.1, 3, 0.8, 0],
        ],
        dtype=np.float32,
    )
    sampled = sample_observer_seconds(fixations, 3, length=4, step_s=1.0)
    assert sampled is not None
    assert sampled.xyz.shape == (4, 3)
    assert sampled.repaired_bins == (1,)
    np.testing.assert_allclose(np.linalg.norm(sampled.xyz, axis=1), 1.0, atol=1e-6)


def test_second_sampling_rejects_long_missing_run() -> None:
    fixations = np.asarray(
        [[0.75, 0.50, 0.1, 1, 0.8, 0], [0.85, 0.60, 3.1, 1, 0.8, 0]], dtype=np.float32
    )
    assert sample_observer_seconds(fixations, 1, length=4, step_s=1.0) is None
