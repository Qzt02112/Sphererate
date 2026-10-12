from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sphererate.data.early_horizon import (
    audit_fixation_sequences,
    build_cumulative_density,
    clipped_dwell_weights,
    load_salient360_eye_fixations,
)
from sphererate.data.maps import MapGeometry


def _write_scanpaths(path: Path) -> None:
    path.write_text(
        "Idx, longitude, latitude, start timestamp\n"
        "0,0.990,0.500,100\n"
        "1,0.010,0.500,600\n"
        "2,0.200,0.400,1400\n"
        "0,0.400,0.500,50\n"
        "1,0.500,0.500,550\n"
        "2,0.600,0.500,1550\n",
        encoding="utf-8",
    )


def test_load_eye_fixations_uses_one_physical_observer_per_sequence_and_seconds(tmp_path: Path) -> None:
    path = tmp_path / "scanpaths.txt"
    _write_scanpaths(path)

    fixations = load_salient360_eye_fixations(path)

    assert sorted(np.unique(fixations[:, 3]).astype(int).tolist()) == [0, 1]
    np.testing.assert_allclose(fixations[fixations[:, 3] == 0, 2], [0.1, 0.6, 1.4])
    np.testing.assert_allclose(fixations[fixations[:, 3] == 0, 4], [0.5, 0.8, 0.65])


def test_clipped_dwell_weights_only_count_overlap_with_horizon() -> None:
    fixations = np.array(
        [
            [0.2, 0.5, 0.1, 0.0, 0.5, 0.0],
            [0.3, 0.5, 0.6, 0.0, 0.8, 0.0],
            [0.4, 0.5, 1.4, 0.0, 0.2, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(clipped_dwell_weights(fixations, 1.0), [0.5, 0.4, 0.0], atol=1e-7)


def test_cumulative_density_wraps_seam_and_has_unit_spherical_integral() -> None:
    fixations = np.array(
        [
            [0.999, 0.5, 0.0, 0.0, 1.0, 0.0],
            [0.001, 0.5, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    density = build_cumulative_density(
        fixations,
        observer_ids=[0, 1],
        horizon_s=0.5,
        height=16,
        width=32,
        sigma_px=1.5,
    )
    area = MapGeometry(16, 32).area_weights

    assert density[:, 0].max() > 0
    assert density[:, -1].max() > 0
    assert float((density * area).sum()) == pytest.approx(1.0, abs=1e-5)


def test_cumulative_density_rejects_observer_half_without_early_mass() -> None:
    fixations = np.array([[0.2, 0.5, 2.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="no positive dwell"):
        build_cumulative_density(
            fixations,
            observer_ids=[0],
            horizon_s=0.5,
            height=8,
            width=16,
            sigma_px=1.0,
        )


def test_sequence_audit_reports_monotonicity_and_horizon_coverage(tmp_path: Path) -> None:
    path = tmp_path / "scanpaths.txt"
    _write_scanpaths(path)
    fixations = load_salient360_eye_fixations(path)

    audit = audit_fixation_sequences(fixations, horizons_s=(0.5, 1.0, 3.0))

    assert audit["observer_count"] == 2
    assert audit["non_monotonic_observer_count"] == 0
    assert audit["observers_with_mass_by_horizon"]["0.5"] == 2
    assert audit["observers_covering_horizon"]["3"] == 0
