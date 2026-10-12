from __future__ import annotations

import numpy as np

from sphererate.evaluation.temporal_factorization import (
    accumulate_temporal_maps,
    align_to_start,
    centered_log_residual,
    deterministic_observer_split,
    restore_from_start,
    select_monotonic_alphas,
    weighted_cosine,
)


def test_start_alignment_round_trip_and_wrap() -> None:
    value = np.arange(16, dtype=np.float32)[None, :]
    for longitude in (-180.0, -90.0, 0.0, 90.0, 180.0):
        aligned = align_to_start(value, longitude)
        restored = restore_from_start(aligned, longitude)
        np.testing.assert_array_equal(restored, value)
    np.testing.assert_array_equal(align_to_start(value, -180.0), align_to_start(value, 180.0))


def test_monotonic_alpha_selection_uses_global_dynamic_programming() -> None:
    alpha_grid = [0.0, 0.5, 1.0]
    costs = np.asarray(
        [
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )
    selected = select_monotonic_alphas(costs, alpha_grid)
    assert selected == [1.0, 0.5, 0.0]


def test_temporal_accumulation_uses_fixed_weights_and_area_normalization() -> None:
    maps = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    area = np.asarray([[1.0, 0.5]], dtype=np.float32)
    final = accumulate_temporal_maps(maps, [0.25, 0.75], area)
    assert np.isclose((final * area).sum(), 1.0)
    assert final[0, 1] > final[0, 0]


def test_centered_log_residual_has_zero_area_weighted_mean() -> None:
    target = np.asarray([[0.8, 0.2]], dtype=np.float32)
    template = np.asarray([[0.5, 0.5]], dtype=np.float32)
    area = np.asarray([[1.0, 0.5]], dtype=np.float32)
    residual = centered_log_residual(target, template, area)
    assert np.isclose(np.average(residual, weights=area), 0.0, atol=1e-7)
    assert np.isclose(weighted_cosine(residual, residual, area), 1.0)


def test_observer_split_is_deterministic_and_preserves_observers() -> None:
    observer_ids = list(range(15))
    left, right = deterministic_observer_split(observer_ids, key="sample", seed=20260711)
    repeated = deterministic_observer_split(observer_ids, key="sample", seed=20260711)
    assert (left, right) == repeated
    assert len(left) == 7
    assert len(right) == 8
    assert sorted(left + right) == observer_ids
    assert not set(left).intersection(right)
