from __future__ import annotations

import numpy as np

from sphererate.baselines.deepgaze_msdb import (
    LogLinearCalibration,
    apply_log_linear_calibration,
    build_train_priors,
    normalize_spherical,
    spherical_area_weights,
)


def test_spherical_normalization_uses_erp_area() -> None:
    area = spherical_area_weights(8, 16)
    density = normalize_spherical(np.ones((8, 16), dtype=np.float32), area)
    assert np.isclose(float((density * area).sum()), 1.0, atol=1e-6)
    assert np.all(np.isfinite(density))


def test_train_prior_is_leave_one_content_out() -> None:
    area = spherical_area_weights(4, 8)
    targets = np.zeros((3, 4, 8), dtype=np.float32)
    targets[0, 1, 1] = 1.0
    targets[1, 1, 3] = 1.0
    targets[2, 2, 5] = 1.0
    full, loo = build_train_priors(targets, area)
    assert full.shape == (4, 8)
    assert loo.shape == targets.shape
    assert loo[0, 1, 1] == 0.0
    assert np.isclose(float((full * area).sum()), 1.0, atol=1e-6)
    assert np.allclose((loo * area[None]).sum(axis=(1, 2)), 1.0, atol=1e-6)


def test_log_linear_calibration_returns_finite_spherical_density() -> None:
    area = spherical_area_weights(6, 12)
    content = np.linspace(0.0, 1.0, 72, dtype=np.float32).reshape(6, 12)
    prior = np.flip(content, axis=1).copy()
    calibration = LogLinearCalibration(
        content_temperature=1.2,
        prior_weight=0.3,
        train_kld_sph=0.0,
        steps=1,
    )
    prediction = apply_log_linear_calibration(content, prior, area, calibration)
    assert prediction.shape == content.shape
    assert np.all(np.isfinite(prediction))
    assert np.isclose(float((prediction * area).sum()), 1.0, atol=1e-6)
