from __future__ import annotations

import numpy as np

from sphererate.baselines.reachability import angular_distance_grid_deg, gate_final_map, weighted_quantile


def test_distance_grid_is_spherical_and_respects_seam() -> None:
    distance = angular_distance_grid_deg(180, 360, start_x_norm=0.0, start_y_norm=0.5)
    assert float(distance[90, 0]) < 1.0
    assert float(distance[90, -1]) < 2.0
    assert float(distance[90, 180]) > 179.0


def test_hard_gate_removes_unreachable_saliency() -> None:
    distance = np.asarray([[0.0, 30.0, 60.0]], dtype=np.float32)
    gated = gate_final_map(np.ones_like(distance), distance, radius_deg=30.0)
    np.testing.assert_array_equal(gated, np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32))


def test_weighted_quantile_uses_dwell_mass() -> None:
    result = weighted_quantile(
        np.asarray([1.0, 2.0, 100.0]),
        np.asarray([1.0, 8.0, 1.0]),
        0.8,
    )
    assert result == 2.0
