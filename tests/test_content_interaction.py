from __future__ import annotations

import numpy as np

from sphererate.evaluation.content_interaction import (
    density_mix,
    deterministic_wrong_image_indices,
    grouped_bootstrap_improvement,
    log_density_mix,
)


def test_density_mix_endpoints_are_area_normalized() -> None:
    start = np.asarray([[0.8, 0.2]], dtype=np.float32)
    static = np.asarray([[0.1, 0.9]], dtype=np.float32)
    area = np.ones_like(start)

    np.testing.assert_allclose(density_mix(start, static, 1.0, area), start, atol=1e-7)
    np.testing.assert_allclose(density_mix(start, static, 0.0, area), static, atol=1e-7)


def test_log_density_mix_is_positive_and_area_normalized() -> None:
    start = np.asarray([[0.8, 0.2]], dtype=np.float32)
    static = np.asarray([[0.1, 0.9]], dtype=np.float32)
    area = np.asarray([[1.0, 0.5]], dtype=np.float32)

    pred = log_density_mix(start, static, alpha=0.5, area=area)

    assert np.all(pred > 0.0)
    assert np.isclose((pred * area).sum(), 1.0)


def test_grouped_bootstrap_orients_lower_and_higher_metrics_as_improvement() -> None:
    candidate = {
        "a1": {"kld_sph": 0.1, "sim_sph": 0.8},
        "a2": {"kld_sph": 0.2, "sim_sph": 0.7},
        "b1": {"kld_sph": 0.3, "sim_sph": 0.6},
    }
    control = {
        "a1": {"kld_sph": 0.2, "sim_sph": 0.7},
        "a2": {"kld_sph": 0.3, "sim_sph": 0.6},
        "b1": {"kld_sph": 0.4, "sim_sph": 0.5},
    }
    groups = {"a1": "a", "a2": "a", "b1": "b"}

    for metric in ("kld_sph", "sim_sph"):
        result = grouped_bootstrap_improvement(candidate, control, groups, metric, iterations=1000, seed=7)
        assert np.isclose(result["oriented_improvement_mean"], 0.1)
        assert result["ci95_low"] > 0.0
        assert result["num_groups"] == 2


def test_wrong_image_mapping_is_deterministic_one_to_one_and_cross_group() -> None:
    rows = [
        {"image_id": "a1", "group_id": "a"},
        {"image_id": "a2", "group_id": "a"},
        {"image_id": "b1", "group_id": "b"},
        {"image_id": "b2", "group_id": "b"},
        {"image_id": "c1", "group_id": "c"},
        {"image_id": "c2", "group_id": "c"},
    ]

    first = deterministic_wrong_image_indices(rows, seed=11)
    repeated = deterministic_wrong_image_indices(rows, seed=11)

    assert first == repeated
    assert sorted(first) == list(range(len(rows)))
    for target_index, donor_index in enumerate(first):
        assert target_index != donor_index
        assert rows[target_index]["group_id"] != rows[donor_index]["group_id"]
