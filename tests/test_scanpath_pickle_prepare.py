from __future__ import annotations

import numpy as np

from sphererate.data.prepare import unit_vectors_to_fixations


def test_unit_vectors_to_fixations_uses_erp_coordinates() -> None:
    scanpaths = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ],
        dtype=np.float32,
    )

    fixations = unit_vectors_to_fixations(scanpaths, num_bins=2)

    assert fixations.shape == (4, 6)
    np.testing.assert_allclose(fixations[:, 0], [0.5, 0.75, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(fixations[:, 1], [0.5, 0.5, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(fixations[:, 5], [0.0, 0.0, 1.0, 1.0], atol=1e-6)
