from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from scripts.train_kernel_v2_pathformer3d import CONFIG, _lr_multiplier, _start_xyz
    from scripts.select_kernel_v2_pathformer3d import _lat_lon_to_normalized_yx
except ModuleNotFoundError as exc:
    # The Pathformer3D adapter imports the upstream checkout, which is not
    # vendored in this repository. Point SCANDMM_ROOT at a checkout containing
    # third_party/Pathformer3D to enable these tests.
    pytest.skip(
        f"Pathformer3D third-party sources unavailable (missing '{exc.name}')",
        allow_module_level=True,
    )


def test_pathformer_uses_frozen_declared_start() -> None:
    np.testing.assert_allclose(
        _start_xyz(torch.device("cpu")).numpy(),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_pathformer_sequence_and_lr_schedule_are_frozen() -> None:
    assert CONFIG["max_length"] == 20
    assert _lr_multiplier(0) == 0.0
    assert _lr_multiplier(10) == 1.0
    assert _lr_multiplier(20) == 0.5
    assert _lr_multiplier(30) == 0.25


def test_pathformer_inverse_coordinates_are_normalized_yx() -> None:
    converted = _lat_lon_to_normalized_yx(
        torch.tensor([[0.0, 90.0], [-90.0, -180.0], [90.0, 180.0]])
    )
    np.testing.assert_allclose(
        converted,
        np.asarray([[0.5, 0.75], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )
