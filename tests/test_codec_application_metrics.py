from __future__ import annotations

import numpy as np

from scripts.evaluate_codec_tile_prefetch import _mixed_psnr, _rank_mass


def test_mixed_quality_improves_when_a_better_tile_is_upgraded() -> None:
    low = np.array([100.0, 50.0])
    high = np.array([10.0, 50.0])
    base = _mixed_psnr(low, high, np.array([False, False]), denominator=2.0)
    upgraded = _mixed_psnr(low, high, np.array([True, False]), denominator=2.0)
    assert upgraded > base


def test_gt_mass_rank_is_stable_for_ties() -> None:
    assert _rank_mass(np.array([[0.2, 0.8], [0.2, 0.0]])).tolist() == [1, 0, 2, 3]
