from __future__ import annotations

import torch

from sphererate.evaluation.temporal_factorization import reference_alignment_pixel_shift


def test_etdk2_start_is_aligned_to_salient360_plus_90_degree_reference() -> None:
    rgb = torch.zeros(3, 4, 16)
    rgb[:, 2, 8] = 1.0  # ET-DK2 start yaw 0 -> x_norm 0.5.

    shift = reference_alignment_pixel_shift(0.0, 90.0, rgb.shape[-1])
    aligned = torch.roll(rgb, shifts=shift, dims=-1)

    assert shift == 4
    assert aligned[:, 2, 12].sum() == 3.0  # Salient360 canonical x_norm 0.75.
    restored = torch.roll(aligned, shifts=-shift, dims=-1)
    assert restored[:, 2, 8].sum() == 3.0
