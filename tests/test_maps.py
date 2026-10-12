from __future__ import annotations

import numpy as np
import pytest

from sphererate.data.maps import (
    MapGeometry,
    build_temporal_maps,
    erp_gaussian_map,
    erp_pixel_center_unit_vectors,
    normalized_coords_to_unit_vectors,
    sitzmann_reference_map,
    spherical_gaussian_map,
    spherical_gaussian_maps,
)
from sphererate.evaluation.metrics import saliency_scores


def test_temporal_maps_shapes_and_normalization() -> None:
    fix = np.array(
        [
            [0.1, 0.5, 0.0, 0.0, 1.0, 0.0],
            [0.9, 0.5, 1.0, 0.0, 1.0, 1.0],
            [0.3, 0.4, 2.0, 1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    maps = build_temporal_maps(fix, num_bins=3, height=16, width=32, sigma_px=2.0)
    assert maps["temporal"].shape == (3, 16, 32)
    assert maps["cumulative"].shape == (3, 16, 32)
    assert maps["final"].shape == (16, 32)
    assert maps["fixation"].sum() == 3
    area = MapGeometry(16, 32).area_weights
    assert np.allclose((maps["temporal"] * area[None]).sum(axis=(1, 2)), 1.0, atol=1e-4)
    assert np.allclose((maps["cumulative"] * area[None]).sum(axis=(1, 2)), 1.0, atol=1e-4)
    assert np.isclose((maps["final"] * area).sum(), 1.0, atol=1e-4)


def test_erp_gaussian_wraps_across_horizontal_seam() -> None:
    height, width = 16, 32
    area = MapGeometry(height, width).area_weights
    seam_map = erp_gaussian_map(
        np.array([0.0], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        height,
        width,
        sigma_px=2.0,
        area_weights=area,
        area_weighted=True,
    )

    middle_row = height // 2
    assert np.isclose(seam_map[middle_row, 1], seam_map[middle_row, -1], rtol=1e-5, atol=1e-7)
    assert seam_map[middle_row, -1] > seam_map[middle_row, width // 2]
    assert np.isclose((seam_map * area).sum(), 1.0, atol=1e-5)


def test_human_style_scores_are_finite() -> None:
    fix_a = np.array([[0.1, 0.5, 0.0, 0.0, 1.0, 0.0], [0.2, 0.5, 1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    fix_b = np.array([[0.12, 0.5, 0.0, 1.0, 1.0, 0.0], [0.22, 0.5, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    maps_a = build_temporal_maps(fix_a, num_bins=2, height=16, width=32, sigma_px=2.0)
    maps_b = build_temporal_maps(fix_b, num_bins=2, height=16, width=32, sigma_px=2.0)
    scores = saliency_scores(maps_a["final"], maps_b["final"], maps_b["fixation"], maps_a["area_weights"]).as_dict()
    assert np.isfinite(scores["kld_sph"])
    assert np.isfinite(scores["sim_sph"])


def test_spherical_gaussian_has_unit_spherical_integral_and_seam_equivariance() -> None:
    height, width = 64, 128
    area = MapGeometry(height, width).area_weights
    first = spherical_gaussian_map(
        np.asarray([0.25], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        height,
        width,
        sigma_deg=4.0,
        area_weights=area,
    )
    shifted = spherical_gaussian_map(
        np.asarray([0.375], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        height,
        width,
        sigma_deg=4.0,
        area_weights=area,
    )

    assert np.isclose((first * area).sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(shifted, np.roll(first, width // 8, axis=1), rtol=2e-5, atol=2e-7)


def test_spherical_gaussian_preserves_angular_scale_across_latitude() -> None:
    height, width = 128, 256
    area = MapGeometry(height, width).area_weights
    grid = erp_pixel_center_unit_vectors(height, width)
    moments = []
    for y_norm in (0.5, 1.0 / 6.0):  # equator and +60 degrees latitude
        density = spherical_gaussian_map(
            np.asarray([0.3], dtype=np.float32),
            np.asarray([y_norm], dtype=np.float32),
            height,
            width,
            sigma_deg=4.0,
            area_weights=area,
        )
        centre = normalized_coords_to_unit_vectors(np.asarray([0.3]), np.asarray([y_norm]))[0]
        angle = np.arccos(np.clip(np.einsum("hwc,c->hw", grid, centre), -1.0, 1.0))
        moments.append(float((density * area * angle * angle).sum()))

    assert moments[1] == pytest.approx(moments[0], rel=0.04)


def test_spherical_gaussian_does_not_wrap_across_the_poles() -> None:
    height, width = 64, 128
    area = MapGeometry(height, width).area_weights
    density = spherical_gaussian_map(
        np.asarray([0.2], dtype=np.float32),
        np.asarray([0.01], dtype=np.float32),
        height,
        width,
        sigma_deg=3.0,
        area_weights=area,
    )

    north_mass = float((density[:8] * area[:8]).sum())
    south_mass = float((density[-8:] * area[-8:]).sum())
    assert north_mass > 0.99
    assert south_mass == pytest.approx(0.0, abs=1e-8)


def test_spherical_gaussian_maps_renders_multiple_weight_sets_in_one_pass() -> None:
    x = np.asarray([0.2, 0.7], dtype=np.float32)
    y = np.asarray([0.4, 0.6], dtype=np.float32)
    weights = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    rendered = spherical_gaussian_maps(x, y, 32, 64, 4.0, weights=weights)
    separate = np.stack(
        [
            spherical_gaussian_map(x[:1], y[:1], 32, 64, 4.0),
            spherical_gaussian_map(x[1:], y[1:], 32, 64, 4.0, weights=np.asarray([2.0])),
        ]
    )

    np.testing.assert_allclose(rendered, separate, rtol=2e-5, atol=2e-7)


def test_sitzmann_reference_renderer_is_probability_normalized() -> None:
    density = sitzmann_reference_map(
        np.asarray([0.2, 0.8], dtype=np.float32),
        np.asarray([0.4, 0.6], dtype=np.float32),
        64,
        128,
        sigma_deg=1.0,
    )

    assert density.shape == (64, 128)
    assert np.isfinite(density).all()
    assert float(density.sum()) == pytest.approx(1.0, abs=1e-6)
