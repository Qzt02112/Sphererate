from __future__ import annotations

import numpy as np

from sphererate.data.maps import MapGeometry
from sphererate.evaluation.late_saliency import (
    build_session_window_map,
    deterministic_wrong_content_map,
    estimate_start_template,
    fuse_log_density,
    restore_start_template,
    yaw_align_map,
)


def test_window_map_excludes_prefix_fixations() -> None:
    fixations = np.array(
        [
            [0.25, 0.5, 2.0, 1.0, 1.0, 0.0],
            [0.75, 0.5, 5.0, 1.0, 1.0, 1.0],
            [0.75, 0.5, 10.0, 1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    geometry = MapGeometry(8, 16)

    late = build_session_window_map(fixations, 5.0, 25.0, geometry, sigma_px=0.5)
    full = build_session_window_map(fixations, 0.0, 25.0, geometry, sigma_px=0.5)

    assert late[4, 12] > late[4, 4]
    assert full[4, 4] > late[4, 4]
    assert np.isclose((late * geometry.area_weights).sum(), 1.0, atol=1e-6)


def test_yaw_alignment_round_trip_for_etdk2_starts() -> None:
    density = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)

    for longitude in (-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0):
        aligned = yaw_align_map(density, longitude)
        restored = restore_start_template(aligned, longitude)
        np.testing.assert_array_equal(restored, density)


def test_start_template_is_mean_of_yaw_aligned_train_targets() -> None:
    geometry = MapGeometry(4, 8)
    centered = np.zeros((4, 8), dtype=np.float32)
    centered[2, 4] = 1.0
    shifted = restore_start_template(centered, 90.0)

    template = estimate_start_template(
        [(centered, 0.0), (shifted, 90.0)],
        geometry.area_weights,
    )

    assert int(np.argmax(template)) == int(np.argmax(centered))
    assert np.isclose((template * geometry.area_weights).sum(), 1.0, atol=1e-6)


def test_log_density_fusion_endpoints() -> None:
    area = np.ones((1, 2), dtype=np.float32)
    start = np.array([[0.8, 0.2]], dtype=np.float32)
    content = np.array([[0.1, 0.9]], dtype=np.float32)

    np.testing.assert_allclose(fuse_log_density(start, content, 1.0, area), start, atol=1e-6)
    np.testing.assert_allclose(fuse_log_density(start, content, 0.0, area), content, atol=1e-6)


def test_wrong_content_mapping_is_seeded_derangement() -> None:
    contents = ["P1", "P2", "P3", "P4"]

    first = deterministic_wrong_content_map(contents, seed=17)
    second = deterministic_wrong_content_map(contents, seed=17)

    assert first == second
    assert set(first) == set(contents)
    assert set(first.values()) == set(contents)
    assert all(target != donor for target, donor in first.items())
