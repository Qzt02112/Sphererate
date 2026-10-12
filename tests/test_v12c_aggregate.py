from __future__ import annotations

from scripts.aggregate_v12c_dwell_rate import _collapse_conditions


def test_collapse_conditions_averages_declared_starts_within_scene() -> None:
    rows = [
        {
            "content_id": "scene-a",
            "horizon_s": 1.0,
            "kld_sph": 1.0,
            "sim_sph": 0.2,
            "cc_sph": 0.3,
            "coverage_5": 0.4,
            "coverage_10": 0.5,
            "coverage_15": 0.6,
            "coverage_20": 0.7,
        },
        {
            "content_id": "scene-a",
            "horizon_s": 1.0,
            "kld_sph": 3.0,
            "sim_sph": 0.4,
            "cc_sph": 0.5,
            "coverage_5": 0.6,
            "coverage_10": 0.7,
            "coverage_15": 0.8,
            "coverage_20": 0.9,
        },
    ]
    collapsed = _collapse_conditions(rows)
    assert len(collapsed) == 1
    assert collapsed[0]["kld_sph"] == 2.0
    assert collapsed[0]["coverage_15"] == 0.7

