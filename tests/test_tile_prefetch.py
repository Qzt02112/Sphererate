from __future__ import annotations

import numpy as np
import pytest

from sphererate.data.maps import MapGeometry, erp_gaussian_map
from sphererate.data.early_horizon import raw_dwell_tile_distribution, raw_dwell_topk_consensus
from sphererate.evaluation.tile_prefetch import (
    aggregate_tile_oracle,
    budget_tile_count,
    decide_eh1,
    evaluate_content_tile_oracle,
    gaze_mass_coverage,
    rank_tiles,
)


def test_budget_rounds_up_and_tile_ranking_breaks_ties_by_index() -> None:
    density = np.ones((8, 16), dtype=np.float32)
    area = np.ones_like(density)

    assert budget_tile_count(8, 16, 5) == 7
    assert budget_tile_count(8, 16, 10) == 13
    assert budget_tile_count(8, 16, 20) == 26
    assert rank_tiles(density, area, tile_rows=8, tile_cols=16, budget_percent=5).tolist() == list(range(7))


def test_tile_ranking_uses_spherical_probability_mass() -> None:
    density = np.zeros((4, 8), dtype=np.float32)
    area = np.ones_like(density)
    density[:2, :2] = 1.0
    density[2:, 6:] = 2.0

    selected = rank_tiles(density, area, tile_rows=2, tile_cols=4, budget_percent=10)

    assert selected.tolist() == [7]


def test_gaze_mass_coverage_clips_horizon_and_wraps_longitude() -> None:
    fixations = np.array(
        [
            [0.999, 0.25, 0.0, 0.0, 1.0, 0.0],
            [0.001, 0.25, 0.0, 0.0, 1.0, 0.0],
            [0.500, 0.75, 0.8, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Grid 2x4: first two fixations land in tile 3 and tile 0; the last is tile 6.
    coverage = gaze_mass_coverage(
        fixations,
        observer_ids=[0],
        horizon_s=1.0,
        selected_tile_ids=[0, 3],
        tile_rows=2,
        tile_cols=4,
    )

    assert coverage == pytest.approx(2.0 / 2.2)

    target = raw_dwell_tile_distribution(
        fixations,
        observer_ids=[0],
        horizon_s=1.0,
        tile_rows=2,
        tile_cols=4,
    )
    assert float(target.ravel()[[0, 3]].sum()) == pytest.approx(coverage)
    assert float(target.sum()) == pytest.approx(1.0)


def test_observer_topk_consensus_is_deterministic_and_bounded() -> None:
    fixations = _synthetic_fixations()
    first = raw_dwell_topk_consensus(
        fixations,
        horizons_s=(0.5, 5.0),
        budgets_percent=(25.0, 50.0),
        tile_rows=1,
        tile_cols=4,
        observer_splits=4,
        seed=7,
        key="P1",
    )
    second = raw_dwell_topk_consensus(
        fixations,
        horizons_s=(0.5, 5.0),
        budgets_percent=(25.0, 50.0),
        tile_rows=1,
        tile_cols=4,
        observer_splits=4,
        seed=7,
        key="P1",
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 2, 1, 4)
    assert np.all((first >= 0.0) & (first <= 1.0))
    np.testing.assert_allclose(first.sum(axis=(-2, -1)), np.array([[1.0, 2.0], [1.0, 2.0]]))


def _synthetic_fixations() -> np.ndarray:
    rows = []
    for observer in range(8):
        rows.extend(
            [
                [0.35, 0.5, 0.0, float(observer), 1.0, 0.0],
                [0.85, 0.5, 4.0, float(observer), 8.0, 0.0],
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def test_content_oracle_same_horizon_covers_more_early_gaze_than_static_and_template() -> None:
    height, width = 8, 16
    area = MapGeometry(height, width).area_weights
    template = erp_gaussian_map(
        np.array([0.85], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        height,
        width,
        sigma_px=0.5,
        area_weights=area,
    )
    rows = evaluate_content_tile_oracle(
        content_id="P1",
        fixations=_synthetic_fixations(),
        templates={0.5: template, 1.0: template, 3.0: template, 5.0: template},
        priors={"uniform": np.ones((height, width), dtype=np.float32)},
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        budgets_percent=(10,),
        height=height,
        width=width,
        sigma_px=0.5,
        tile_rows=1,
        tile_cols=4,
        observer_splits=2,
        seed=9,
    )

    assert len(rows) == 4
    assert rows[0]["same_vs_static_improvement"] > 0.9
    assert rows[0]["same_vs_template_improvement"] > 0.9


def test_tile_oracle_gate_uses_content_bootstrap_and_outer_test_guard() -> None:
    rows = []
    for content in range(10):
        for horizon in (0.5, 1.0, 3.0, 5.0):
            rows.append(
                {
                    "content_id": f"P{content}",
                    "horizon_s": horizon,
                    "budget_percent": 10.0,
                    "same_vs_static_improvement": 0.1,
                    "same_vs_template_improvement": 0.05,
                }
            )
    aggregates = aggregate_tile_oracle(rows, primary_budget_percent=10, iterations=1000, seed=5)
    decision = decide_eh1(aggregates, outer_test_used=False)

    assert decision["decision"] == "PASS"
    assert decide_eh1(aggregates, outer_test_used=True)["decision"] == "FAIL"
